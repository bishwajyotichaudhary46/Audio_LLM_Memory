import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List

import io
import soundfile as sf
import torch
import evaluate

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    WhisperForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType



# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "openai/whisper-small"
DATA_PATH = "Data/indicvoice"
OUTPUT_DIR = "artifacts/checkpoint/lora"

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
NUM_EPOCHS = 50
MAX_LABEL_LENGTH = 448
GEN_MAX_LENGTH = 225
SEED = 42

# LoRA hyperparameters
LORA_RANK    = 32
LORA_ALPHA   = 64       # α = 2 × rank
LORA_DROPOUT = 0.05

print("Using device:", DEVICE)



# LOAD PROCESSOR / TOKENIZER

processor = WhisperProcessor.from_pretrained(BASE_MODEL)

tokenizer_transcribe = WhisperTokenizer.from_pretrained(
    BASE_MODEL,
    language="ne",
    task="transcribe",
)



# LOAD BASE MODEL

base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)

# NOTE: proj_out / tie_word_embeddings block removed — those were specific to
#       the custom WhisperCustom class and do not exist on the standard model.
base_model.config.use_cache = False

base_model.generation_config.language = "ne"
base_model.generation_config.task = "transcribe"
base_model.generation_config.forced_decoder_ids = None
base_model.generation_config.suppress_tokens = []
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []



# WRAP WITH LORA

# Standard Whisper has no custom memory layers, so modules_to_save is empty.
# We target q_proj + v_proj in both encoder and decoder attention blocks.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    # modules_to_save is intentionally omitted — no custom layers here
)

model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.print_trainable_parameters()



# LOAD DATASET
dataset = load_from_disk(DATA_PATH)
dataset = dataset.map(lambda x: {"lang": ["ne"] * len(x["text"])}, batched=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset["train"] = dataset["train"].shuffle(seed=SEED)

print("Train size:", len(dataset["train"]))
print("Val size  :", len(dataset["valid"]))



# DATA COLLATOR

@dataclass
class DataCollatorSpeechSeq2SeqTranscriptionOnly:
    processor: Any
    tokenizer: Any
    decoder_start_token_id: int
    max_label_length: int = 448

    def _decode_audio(self, audio_obj, idx: int):
        if hasattr(audio_obj, "get_all_samples"):
            try:
                samples = audio_obj.get_all_samples()
                audio_array = samples.data
                sr = samples.sample_rate
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()
                if audio_array.ndim == 2:
                    audio_array = audio_array.mean(axis=0) if audio_array.shape[0] <= 8 else audio_array.mean(axis=1)
                return audio_array.astype("float32"), sr
            except Exception as e:
                raise ValueError(f"AudioDecoder failed at index {idx}: {e}")

        if isinstance(audio_obj, dict) and "array" in audio_obj:
            audio_array = audio_obj["array"]
            sr = audio_obj.get("sampling_rate", 16000)
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)
            return audio_array.astype("float32"), sr

        if isinstance(audio_obj, dict) and "bytes" in audio_obj:
            try:
                buffer = io.BytesIO(audio_obj["bytes"])
                audio_array, sr = sf.read(buffer)
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=-1)
                return audio_array.astype("float32"), sr
            except Exception as e:
                raise ValueError(f"Bytes decode failed at index {idx}: {e}")

        raise ValueError(f"Unsupported audio type at index {idx}: {type(audio_obj)}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays, valid_features = [], []

        for i, f in enumerate(features):
            if "audio" not in f:
                print(f"Skipping sample {i}: missing audio")
                continue
            try:
                audio_array, sr = self._decode_audio(f["audio"], i)
                if sr != 16000:
                    print(f"Skipping sample {i}: expected 16000 Hz, got {sr}")
                    continue
                audio_arrays.append(audio_array)
                valid_features.append(f)
            except Exception as e:
                print(f"Skipping bad audio at index {i}: {e}")
                continue

        if not audio_arrays:
            raise ValueError("All audio samples in this batch failed to decode.")

        batch_inputs = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        )

        labels = [
            {"input_ids": self.tokenizer(
                f["text"],
                truncation=True,
                max_length=self.max_label_length,
                return_tensors=None,
            )["input_ids"]}
            for f in valid_features
        ]

        labels_batch = self.tokenizer.pad(labels, padding=True, return_tensors="pt")
        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if labels_tensor.size(1) > 0 and (labels_tensor[:, 0] == self.decoder_start_token_id).all():
            labels_tensor = labels_tensor[:, 1:]

        return {
            "input_features": batch_inputs["input_features"],
            "labels": labels_tensor,
        }


data_collator = DataCollatorSpeechSeq2SeqTranscriptionOnly(
    processor=processor,
    tokenizer=tokenizer_transcribe,
    decoder_start_token_id=model.config.decoder_start_token_id,
    max_label_length=MAX_LABEL_LENGTH,
)



wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer_transcribe.pad_token_id

    pred_str  = [normalize_text(s) for s in tokenizer_transcribe.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_text(s) for s in tokenizer_transcribe.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }



training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    generation_max_length=GEN_MAX_LENGTH,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    report_to=["tensorboard"],
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)




# Whisper forward() only accepts these keys — anything else (e.g. input_ids
# injected by the Trainer/PEFT layer) must be stripped before the forward pass.
WHISPER_FORWARD_KEYS = {
    "input_features",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "head_mask",
    "decoder_head_mask",
    "cross_attn_head_mask",
    "encoder_outputs",
    "past_key_values",
    "decoder_inputs_embeds",
    "labels",
    "use_cache",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}

class DebugTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Strip any keys Whisper doesn't understand (e.g. input_ids from PEFT)
        inputs = {k: v for k, v in inputs.items() if k in WHISPER_FORWARD_KEYS}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = {k: v for k, v in inputs.items() if k in WHISPER_FORWARD_KEYS}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def save_model(self, output_dir=None, _internal_call=False):
        state_dict = self.model.state_dict()
        for name, _ in self.model.named_parameters():
            if name not in state_dict:
                print(f"Missing key in state_dict: {name}")
        super().save_model(output_dir, _internal_call)

    def evaluation_loop(self, *args, **kwargs):
        torch.cuda.empty_cache()
        result = super().evaluation_loop(*args, **kwargs)
        torch.cuda.empty_cache()
        return result



trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.00001,
        )
    ],
)



trainer.train()


adapter_dir = os.path.join(OUTPUT_DIR, "best_model", "lora_adapter")
model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)
tokenizer_transcribe.save_pretrained(os.path.join(adapter_dir, "tokenizer_transcribe"))
print(f"LoRA adapter saved → {adapter_dir}")



print("Merging LoRA weights into base model …")
merged_model = model.merge_and_unload()

merged_dir = os.path.join(OUTPUT_DIR, "best_model", "merged")
merged_model.save_pretrained(merged_dir)
processor.save_pretrained(merged_dir)
tokenizer_transcribe.save_pretrained(os.path.join(merged_dir, "tokenizer_transcribe"))
print(f"Merged model saved → {merged_dir}")



# INFERENCE with merged model (no PEFT needed)

# import librosa
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
#
# proc    = WhisperProcessor.from_pretrained(merged_dir)
# model_  = WhisperForConditionalGeneration.from_pretrained(merged_dir).to("cuda")
# speech, sr = librosa.load("sample.wav", sr=16000)
# inputs  = proc(speech, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
# with torch.no_grad():
#     ids = model_.generate(inputs)
# print(proc.batch_decode(ids, skip_special_tokens=True)[0])