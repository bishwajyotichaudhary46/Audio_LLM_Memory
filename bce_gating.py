"""
train_bce_sft.py
================
SFT training for Whisper + Titans (last_layer_gating_glu) on Nepali speech.

Architecture
------------
Each training step has two backward passes:
  1. Gate pass  — BCE loss on router_gate/router_value/router_proj params only.
                  Label: 1 if argmax prediction ≠ reference, 0 if correct.
  2. Memory pass — CE (cross-entropy) loss on mem_block params only.

Using HuggingFace Trainer infrastructure (data collation, evaluation,
checkpointing, early stopping, TensorBoard) but a fully custom training_step.

Fixes over the original
-----------------------
* freeze_parameters / unfreeze_parameters names now match their behaviour.
* Gate label derived from argmax logits (fast, no generate() call needed for
  a binary signal).
* Only ONE backward per loss; GradScaler is used for the CE pass only (the
  gate pass is fp32 tensors after .float() cast to avoid mixed-precision NaN).
* best_wer variable correctly tracks WER (was called best_bleu).
* Duplicate normalize_text / metric definitions removed.
* Gradient clipping applied once, after unscaling.
"""

import os
import re
import io
import json
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
import torch.nn.functional as F
import evaluate
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning,   module="transformers")

from datasets import Audio, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    get_scheduler,
)

from MemoryModule.memory.last_layer_gating_glu import (
    WhisperForConditionalGeneration as WhisperCustom,
)



# CONFIG

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

MODEL_NAME = "openai/whisper-small"
DATA_PATH  = "Data/indicvoice"
OUTPUT_DIR = "artifacts/transcribe/bce_sft/last_layer_gating_glu"

LANGUAGE = "ne"
TASK     = "transcribe"

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 4
GRAD_ACCUM_STEPS = 8       
MAX_LABEL_LEN    = 448
GEN_MAX_LENGTH   = 225

SFT_EPOCHS       = 20
SFT_LR           = 5e-5
GATE_LR          = 1e-4    # gates typically benefit from a slightly higher LR
SFT_WEIGHT_DECAY = 0.01
SFT_WARMUP_RATIO = 0.1
PATIENCE         = 2
SEED             = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")



# PROCESSOR / TOKENIZER
processor  = WhisperProcessor.from_pretrained(MODEL_NAME)
feature_extractor = processor.feature_extractor

tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

print(f"Decoder start token id : {processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')}")



# MODEL
model = WhisperCustom.from_pretrained(MODEL_NAME).to(DEVICE)
model.config.use_cache            = False
model.generation_config.language  = LANGUAGE
model.generation_config.task      = TASK
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens    = []
model.config.forced_decoder_ids   = None
model.config.suppress_tokens      = []

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# Tie output embeddings (saves ~10 M params, keeps generation coherent)
model.config.tie_word_embeddings = False
model.proj_out.weight = torch.nn.Parameter(
    model.model.decoder.embed_tokens.weight.detach().clone()
)

# GradScaler: needed for fp16, harmless no-op for bf16
scaler = GradScaler(enabled=(AMP_DTYPE == torch.float16))



# TRAINABLE PARAMETERS
GATE_KEYWORDS   = ["router_gate", "router_value", "router_proj"]
MEMORY_KEYWORDS = ["mem_block"]


def set_requires_grad(model, keywords, value: bool):
    for name, param in model.named_parameters():
        if any(k in name.lower() for k in keywords):
            param.requires_grad = value


# Start with everything frozen; we toggle per-pass inside the training step.
for p in model.parameters():
    p.requires_grad = False

# Initialise gate projection bias negative so gates start near-closed.
for name, module in model.named_modules():
    if "router_proj" in name.lower():
        if hasattr(module, "weight") and module.weight is not None:
            torch.nn.init.xavier_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.constant_(module.bias, -4.0)
        print(f"Initialized gate projection: {name}")

print("\nTrainable gate parameters:")
set_requires_grad(model, GATE_KEYWORDS, True)
set_requires_grad(model, MEMORY_KEYWORDS, True)
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  {name}")
# Freeze again — toggled during training step
for p in model.parameters():
    p.requires_grad = False

num_total     = sum(p.numel() for p in model.parameters())
num_gate      = sum(p.numel() for n, p in model.named_parameters() if any(k in n.lower() for k in GATE_KEYWORDS))
num_memory    = sum(p.numel() for n, p in model.named_parameters() if any(k in n.lower() for k in MEMORY_KEYWORDS))
print(f"\nTotal params  : {num_total:,}")
print(f"Gate params   : {num_gate:,}")
print(f"Memory params : {num_memory:,}")



# DATASET
print("\nLoading dataset …")
dataset = load_from_disk(DATA_PATH)
dataset.cleanup_cache_files()
dataset = dataset.map(lambda x: {"lang": ["ne"] * len(x["text"])}, batched=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

train_raw = dataset["train"]
val_raw   = dataset["valid"]
print(f"Train : {len(train_raw):,}  |  Val : {len(val_raw):,}")



# DATA COLLATOR
@dataclass
class DataCollatorWhisperBCE:
    processor: Any
    tokenizer: Any
    decoder_start_token_id: int
    max_label_length: int = 448

    @staticmethod
    def _load_audio(audio_obj, idx):
        if hasattr(audio_obj, "get_all_samples"):
            s = audio_obj.get_all_samples()
            arr = s.data
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            if arr.ndim == 2:
                arr = arr.mean(axis=0 if arr.shape[0] <= 8 else 1)
            return arr.astype("float32")
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            arr = audio_obj["array"]
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype("float32")
        if isinstance(audio_obj, dict) and "bytes" in audio_obj:
            arr, _ = sf.read(io.BytesIO(audio_obj["bytes"]))
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.astype("float32")
        raise ValueError(f"Unsupported audio type at index {idx}: {type(audio_obj)}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays, valid = [], []
        for i, f in enumerate(features):
            try:
                audio_arrays.append(self._load_audio(f["audio"], i))
                valid.append(f)
            except Exception as e:
                print(f"  [collate] skipping sample {i}: {e}")

        if not audio_arrays:
            raise ValueError("All audio samples in batch failed to decode.")

        feats = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        )
        labels = []
        for f in valid:
            tok = self.tokenizer(
                f["text"],
                truncation=True,
                max_length=self.max_label_length,
                return_tensors=None,
            )
            labels.append({"input_ids": tok["input_ids"]})

        labels_batch  = self.tokenizer.pad(labels, padding=True, return_tensors="pt")
        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        # Strip leading decoder-start token (Whisper adds it internally)
        if (
            labels_tensor.size(1) > 0
            and (labels_tensor[:, 0] == self.decoder_start_token_id).all()
        ):
            labels_tensor = labels_tensor[:, 1:]

        return {
            "input_features": feats["input_features"],
            "labels": labels_tensor,
        }


data_collator = DataCollatorWhisperBCE(
    processor=processor,
    tokenizer=tokenizer,
    decoder_start_token_id=model.config.decoder_start_token_id,
    max_label_length=MAX_LABEL_LEN,
)



# METRICS
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = [normalize_text(s) for s in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_text(s) for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }



# OPTIMIZERS  (one per parameter group)

def build_optimizer(keywords, lr):
    set_requires_grad(model, keywords, True)
    params = [p for n, p in model.named_parameters() if any(k in n.lower() for k in keywords)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=SFT_WEIGHT_DECAY)
    set_requires_grad(model, keywords, False)
    return opt


gate_optimizer   = build_optimizer(GATE_KEYWORDS,   GATE_LR)
memory_optimizer = build_optimizer(MEMORY_KEYWORDS, SFT_LR)



# BCE GATE LABEL  (fast: uses argmax logits, no generate())

bce_loss_fn = torch.nn.BCEWithLogitsLoss()


def gate_labels_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Returns a float tensor of shape (B,) with:
      0  — argmax prediction matches reference for ALL non-padding tokens
      1  — at least one token is wrong  (model should route through memory)
    """
    pred = logits.argmax(dim=-1)          # (B, T)
    mask = labels.ne(-100)                 # ignore padding
    correct = ((pred == labels) | ~mask).all(dim=-1)   # (B,)
    return (~correct).float()



# CUSTOM TRAINER  (two-pass BCE + CE training step)

class BCESFTTrainer(Seq2SeqTrainer):
    """
    Overrides training_step to perform:
      Pass 1 — gate params only, BCE loss on router correctness signal
      Pass 2 — memory params only, CE loss (standard Whisper transcription)
    """

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        feats  = inputs["input_features"].to(DEVICE)
        labels = inputs["labels"].to(DEVICE)


        set_requires_grad(model, GATE_KEYWORDS, True)

        with autocast(device_type="cuda", dtype=AMP_DTYPE):
            output = model(input_features=feats, labels=labels)

        bce_logits = getattr(output, "bce_logits", None)

        print(f"  BCE logits shape: {bce_logits.shape if bce_logits is not None else 'N/A'}")
        if bce_logits is not None:
            with torch.no_grad():
                gate_lbl = gate_labels_from_logits(
                    output.logits.detach(), labels
                ).to(DEVICE)

            print(f"Gate labels distribution: {gate_lbl.float().mean().item():.4f} positive (should route through memory)")
            # Gate logits are mean-pooled over sequence before BCE
            gate_pred = bce_logits.mean(dim=1).squeeze(-1).float()   # (B,)
            gate_loss = bce_loss_fn(gate_pred, gate_lbl)
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            gate_optimizer.step()

        set_requires_grad(model, GATE_KEYWORDS, False)
        set_requires_grad(model, MEMORY_KEYWORDS, True)

        with autocast(device_type="cuda", dtype=AMP_DTYPE):
            output = model(input_features=feats, labels=labels)
            ce_loss = output.loss

        memory_optimizer.zero_grad(set_to_none=True)
        scaler.scale(ce_loss).backward()
        scaler.unscale_(memory_optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(memory_optimizer)
        scaler.update()

        set_requires_grad(model, MEMORY_KEYWORDS, False)

        return ce_loss.detach()

    def save_model(self, output_dir=None, _internal_call=False):
        # Temporarily re-enable all trainable params so state_dict is complete
        set_requires_grad(model, GATE_KEYWORDS,   True)
        set_requires_grad(model, MEMORY_KEYWORDS, True)
        super().save_model(output_dir, _internal_call)
        set_requires_grad(model, GATE_KEYWORDS,   False)
        set_requires_grad(model, MEMORY_KEYWORDS, False)

    def evaluation_loop(self, *args, **kwargs):
        torch.cuda.empty_cache()
        result = super().evaluation_loop(*args, **kwargs)
        torch.cuda.empty_cache()
        return result



# TRAINING ARGUMENTS

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=SFT_LR,           # used by HF Trainer's scheduler; we override step
    weight_decay=SFT_WEIGHT_DECAY,
    warmup_ratio=SFT_WARMUP_RATIO,
    num_train_epochs=SFT_EPOCHS,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available() and AMP_DTYPE == torch.float16,
    bf16=torch.cuda.is_available() and AMP_DTYPE == torch.bfloat16,
    predict_with_generate=True,
    generation_max_length=GEN_MAX_LENGTH,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,
    report_to=["tensorboard"],
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    seed=SEED,
)



# TRAINER

trainer = BCESFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_raw,
    eval_dataset=val_raw,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=PATIENCE,
            early_stopping_threshold=1e-4,
        )
    ],
)



# TRAIN
print(f"\n{'='*60}")
print(f"  BCE-SFT  ({SFT_EPOCHS} epochs max, patience={PATIENCE})")
print(f"  Gate LR  : {GATE_LR}   Memory LR : {SFT_LR}")
print(f"  Batch    : {TRAIN_BATCH_SIZE} × {GRAD_ACCUM_STEPS} accum = {TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS} eff.")
print(f"{'='*60}\n")

trainer.train(resume_from_checkpoint=False)

# Save best model + processor
best_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_dir)
processor.save_pretrained(best_dir)
tokenizer.save_pretrained(os.path.join(best_dir, "tokenizer_transcribe"))
print(f"\nBest model saved → {best_dir}")