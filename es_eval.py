import re
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Audio, load_from_disk
from transformers import (
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration
)
from MemoryModule.memory.last_layer_gating_glu import (
    WhisperForConditionalGeneration as WhisperCustom
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMPORTANT:
# Use the SAME processor family as the model/checkpoint family.
# If your checkpoint was trained from whisper-small, use whisper-small.

base_model_name = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(base_model_name)

model = WhisperForConditionalGeneration.from_pretrained("artifacts/transcribe/full_tunning/dpo/checkpoint-20000").to(device)
# model = WhisperCustom.from_pretrained(
#    "artifacts/transcribe/glu/sft/last_layer_gating_glu/best_model"
# ).to(device)





model.eval()


# Whisper generation settings
# Better to set generation_config for language/task

model.generation_config.language = "es"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.suppress_tokens = []


# Dataset

dataset = load_from_disk("Data/hf_eval_es")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def preprocess(batch):
    audio = batch["audio"]["array"]

    # input_features: shape [80, 3000] usually after padding/truncation logic in processor
    batch["input_features"] = processor.feature_extractor(
        audio,
        sampling_rate=16000
    ).input_features[0]

    # labels
    batch["labels"] = processor.tokenizer(
        batch["transcription"]
    ).input_ids

    return batch

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Encoder padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Decoder padding
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
        )

        # Remove decoder start token if it was already appended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# Metric
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")



normalizer = BasicTextNormalizer(remove_diacritics=False, split_letters=False)
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()

    # Replace -100 so tokenizer can decode labels
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True
    )
    label_str = processor.tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True
    )

    pred_str = [normalizer(s) for s in pred_str]
    label_str = [normalizer(s) for s in label_str]

    wer = 100 * wer_metric.compute(
        predictions=pred_str,
        references=label_str
    )
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer,
            "cer": cer,}




# print("\n Models", model )

# Evaluation args
training_args = Seq2SeqTrainingArguments(
    output_dir="eval_result",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=1,
    fp16=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,  # common Whisper trainer setup
)

results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)
