import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List

import io
import soundfile as sf
import torch
import evaluate

from datasets import load_from_disk, Audio, concatenate_datasets, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from MemoryModule.conponents.TitansLastLayerModelingNew import (
    WhisperForConditionalGeneration as WhisperCustom,
)


# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "openai/whisper-small"
DATA_PATH = "Data/indicvoice"
OUTPUT_DIR = "artifacts/transcribe/memory/last_layer/sft/memory_layer_2_gate"

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
NUM_EPOCHS = 20
MAX_LABEL_LENGTH = 448
GEN_MAX_LENGTH = 225
SEED = 42

print("Using device:", DEVICE)


# LOAD PROCESSOR / TOKENIZER
# Transcription setting:
# source speech = Nepali
# target text   = Nepali transcription
# task          = transcribe
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

tokenizer_transcribe = WhisperTokenizer.from_pretrained(
    BASE_MODEL,
    language="ne",
    task="transcribe",
)


# LOAD MODEL
model = WhisperCustom.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
).to(DEVICE)

model.config.tie_word_embeddings = False
model.proj_out.weight = torch.nn.Parameter(
    model.model.decoder.embed_tokens.weight.detach().clone()
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Generation must also be Nepali transcription, not English translation.
model.generation_config.language = "ne"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# LOAD DATASET
# Expected main dataset columns:
# ['chunked_audio_filepath', 'text', 'en_text']
# For transcription, use 'text' as Nepali label, NOT 'en_text'.
dataset = load_from_disk(DATA_PATH)



dataset = dataset.map(lambda x: {"lang": ["ne"] * len(x["text"])}, batched=True)
print(dataset)





# CAST TO AUDIO
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print("Audio column cast complete.")

print(dataset)
# TRAIN / VAL SPLIT
dataset = dataset.shuffle(seed=SEED)




dataset['train'] = dataset['train'].shuffle(seed=SEED)

print("Train raw size:", len(dataset['train']))
print("Val raw size:", len(dataset['valid']))


# DATA COLLATOR
@dataclass
class DataCollatorSpeechSeq2SeqTranscriptionOnly:
    processor: Any
    tokenizer: Any
    decoder_start_token_id: int
    max_label_length: int = 448

    def _decode_audio(self, audio_obj, idx: int):
        # Case 1: Hugging Face / torchcodec AudioDecoder
        if hasattr(audio_obj, "get_all_samples"):
            try:
                samples = audio_obj.get_all_samples()
                audio_array = samples.data
                sr = samples.sample_rate

                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()

                # torchcodec may return (channels, time)
                if audio_array.ndim == 2:
                    if audio_array.shape[0] <= 8:  # likely channels first
                        audio_array = audio_array.mean(axis=0)
                    else:  # likely time x channels
                        audio_array = audio_array.mean(axis=1)

                return audio_array.astype("float32"), sr
            except Exception as e:
                raise ValueError(f"Failed to decode AudioDecoder at batch index {idx}: {e}")

        # Case 2: dict with array
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            audio_array = audio_obj["array"]
            sr = audio_obj.get("sampling_rate", 16000)

            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)

            return audio_array.astype("float32"), sr

        # Case 3: dict with bytes
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
                raise ValueError(f"Failed to decode audio bytes at batch index {idx}: {e}")

        raise ValueError(f"Unsupported audio type at batch index {idx}: {type(audio_obj)}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = []
        valid_features = []

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
                print("\nSkipping bad audio sample")
                print("Batch index:", i)
                print("Audio path:", f.get("audio_path", "NO_PATH_FOUND"))
                print("Error:", e)
                continue

        if len(audio_arrays) == 0:
            raise ValueError("All audio samples in this batch failed to decode.")

        batch_inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )

        labels = []
        for f in valid_features:
            tokenized = self.tokenizer(
                f["text"],
                truncation=True,
                max_length=self.max_label_length,
                return_tensors=None
            )
            labels.append({"input_ids": tokenized["input_ids"]})

        labels_batch = self.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt"
        )

        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
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


# NORMALIZATION
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# METRICS
# For transcription, WER/CER are more suitable than BLEU.
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()

    # Replace -100 with pad token before decoding.
    label_ids[label_ids == -100] = tokenizer_transcribe.pad_token_id

    pred_str = tokenizer_transcribe.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )
    label_str = tokenizer_transcribe.batch_decode(
        label_ids,
        skip_special_tokens=True,
    )

    pred_str = [normalize_text(s) for s in pred_str]
    label_str = [normalize_text(s) for s in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "cer": cer,
    }


# FREEZE / UNFREEZE
# Freeze all, then unfreeze your custom memory/router layers.
for p in model.parameters():
    p.requires_grad = False

for name, module in model.named_modules():
    lname = name.lower()

    if "router_proj" in lname :
        if hasattr(module, "weight") and module.weight is not None:
            torch.nn.init.zeros_(module.weight)

        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.constant_(module.bias, -4.0)

        print(f"Initialized {name}: weight=0, bias=-4")

TRAINABLE_KEYWORDS = [
    "mem_block",
    "router_proj",
    "alpha"
]

for name, param in model.named_parameters():
    lname = name.lower()
    if any(key in lname for key in TRAINABLE_KEYWORDS):
        param.requires_grad = True
        print(f"Trainable: {name}")

print("\nTrainable parameters:")
num_total = 0
num_trainable = 0
for name, param in model.named_parameters():
    num_total += param.numel()
    if param.requires_grad:
        num_trainable += param.numel()
        print(name)

print(f"\nTotal params     : {num_total:,}")
print(f"Trainable params : {num_trainable:,}")

if num_trainable == 0:
    print("\nWARNING: No trainable parameters found.")
    print("Check your custom module parameter names carefully.")


# TRAINING ARGUMENTS
# For transcription, select best checkpoint by lowest WER.
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


# DEBUG TRAINER
class DebugTrainer(Seq2SeqTrainer):
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


# TRAINER
trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.00001,
        )
    ],
)


# TRAIN
trainer.train()
# trainer.evaluate()


# SAVE BEST MODEL
best_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_dir)
processor.save_pretrained(best_dir)
tokenizer_transcribe.save_pretrained(os.path.join(best_dir, "tokenizer_transcribe"))
