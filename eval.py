import os
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    WhisperForConditionalGeneration
)

# Custom Whisper memory model
# from MemoryModule.conponents.context_mom_modeling import (
#     WhisperForConditionalGeneration as WhisperCustom
# )

# from MemoryModule.conponents.only_projection_modeling import (
#     WhisperForConditionalGeneration as WhisperCustom
# )
from MemoryModule.conponents.TitansLastLayerModelingNew import (
    WhisperForConditionalGeneration as WhisperCustom,
)

# Device (CPU)
device = torch.device("cuda")


# Load Whisper components
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model_mem = WhisperForConditionalGeneration.from_pretrained("whisper_translation_full_tunning/checkpoint-36315").to(device)
model_mem = WhisperCustom.from_pretrained("artifacts/transcribe/memory/memory_3_last_layer/dpo/best_model", local_files_only=True).to(device)
# Disable weight tying so proj_out is saved as an independent param
model_mem.config.tie_word_embeddings = False
model_mem.proj_out.weight = torch.nn.Parameter(
    model_mem.model.decoder.embed_tokens.weight.detach().clone()
)
# if os.path.exists("open_slr_memory_model.pt"):
#     state_dict = torch.load("open_slr_memory_model.pt", map_location=device)
#     model_mem.load_state_dict(state_dict, strict=False)

# Load CSV dataset
dataset = load_from_disk("artifacts/Data/eval_en")



# Cast audio column
dataset = dataset.cast_column(
    "audio_path",
    Audio(sampling_rate=16000)
)

print(dataset)

# Preprocessing
def preprocess(batch):
    audio = batch["audio_path"]["array"]

    batch["input_features"] = feature_extractor(
        audio,
        sampling_rate=16000
    ).input_features[0]

    batch["labels"] = tokenizer(
        batch["transcript"]
    ).input_ids

    return batch

# IMPORTANT: operate on split, not DatasetDict
dataset = dataset['train'].map(
    preprocess,
    remove_columns=dataset['train'].column_names
)

print(dataset)


# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features):
        # ---- Encoder inputs ----
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]

        batch = self.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Explicit attention mask (VERY IMPORTANT for Whisper)
        batch["attention_mask"] = torch.ones(
            batch["input_features"].shape[:-1],
            dtype=torch.long
        )

        # Decoder labels 
        label_features = [
            {"input_ids": f["labels"]} for f in features
        ]

        labels_batch = self.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    decoder_start_token_id=model_mem.config.decoder_start_token_id,
)


# Generation config

gen_cfg = GenerationConfig.from_pretrained("openai/whisper-small")
gen_cfg.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="en",
    task="transcribe"
)
gen_cfg.suppress_tokens = []
gen_cfg.suppress_blank = False
gen_cfg.num_beams = 5
gen_cfg.return_timestamps = False

model_mem.generation_config = gen_cfg

# WER Metric
metric = evaluate.load("wer")
import re
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    label_str = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )

    pred_str = [normalize(s) for s in pred_str]
    label_str = [normalize(s) for s in label_str]

    wer = 100 * metric.compute(
        predictions=pred_str,
        references=label_str
    )

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="eval_result",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=False,
    no_cuda=False,
    report_to=["tensorboard"],
    evaluation_strategy="steps",  
    logging_strategy="steps",  
    greater_is_better=False,
    logging_steps=1,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model_mem,
    args=training_args,
    eval_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)
# Run Evaluation
results = trainer.evaluate()
print(results)