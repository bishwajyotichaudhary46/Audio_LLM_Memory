"""
train_ewc.py
─────────────────────────────────────────────────────────────────────────────
Fine-tune Whisper with Elastic Weight Consolidation (EWC) regularisation.

Total loss = CE loss (ASR) + λ * EWC penalty
EWC penalty = Σ_i  F_i * (θ_i - θ*_i)²   (summed over all parameters)

Requires:
  - ewc_fisher_whisper.pt   (produced by compute_fisher_ewc.py)
  - Dataset on disk at DATASET_PTH (load_from_disk), with columns:
    ['chunked_audio_filepath', 'text', 'en_text']  -> 'text' is Nepali label
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import io
import re
import random
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Dict

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import evaluate

from torch.utils.data import DataLoader
from datasets import load_from_disk, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)




try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False



# CONFIG

MODEL_NAME       = "openai/whisper-small"
DATASET_PTH      = "Data/indicvoice"
EWC_PATH         = "artifacts/transcribe/EWC/ewc_fisher_whisper.pt"
CHECKPOINT_DIR   = "artifacts/transcribe/EWC/checkpoints_ewc"
OUTPUT_DIR       = "whisper_ewc_finetuned"

EWC_LAMBDA       = 500      # EWC regularisation strength — tune this!
                             # too low  → catastrophic forgetting
                             # too high → model can't learn new task

BATCH_SIZE       = 8
GRAD_ACCUM_STEPS = 16         # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS       = 30
LEARNING_RATE    = 1e-5
WARMUP_STEPS     = 100
MAX_GRAD_NORM    = 1.0
SAMPLING_RATE    = 16_000
MAX_LABEL_LENGTH = 448
LOG_EVERY        = 500       # optimiser steps

# Early stopping
EARLY_STOPPING_PATIENCE = 3   # epochs without improvement before stopping
EARLY_STOPPING_METRIC   = "wer"  # lower is better

SEED = 42



# REPRODUCIBILITY

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False



# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



# LOAD MODEL + PROCESSORS

print(f"Loading model: {MODEL_NAME}")

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME)
processor         = WhisperProcessor.from_pretrained(MODEL_NAME)
model             = WhisperForConditionalGeneration.from_pretrained("artifacts/transcribe/EWC/checkpoints_ewc/epoch_5").to(device)
model.gradient_checkpointing_enable()
model.train()



# LOAD EWC DATA  (Fisher + θ*)

print(f"Loading EWC data from: {EWC_PATH}")

ewc_data   = torch.load(EWC_PATH, map_location=device)
fisher     = {k: v.to(device) for k, v in ewc_data["fisher"].items()}
theta_star = {k: v.to(device) for k, v in ewc_data["theta_star"].items()}

print(f"  Loaded Fisher for {len(fisher)} parameter tensors.")


def ewc_penalty(model: nn.Module) -> torch.Tensor:
    """
    Computes:  Σ_i  F_i * (θ_i - θ*_i)²
    Returns a scalar tensor (requires_grad=True via current θ_i).
    """
    penalty = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in fisher and name in theta_star:
            penalty = penalty + (
                fisher[name] * (param - theta_star[name]) ** 2
            ).sum()
    return penalty



# LOAD DATASET
print("Loading training dataset...")

# Expected main dataset columns:
# ['chunked_audio_filepath', 'text', 'en_text']
# For transcription, use 'text' as Nepali label, NOT 'en_text'.
dataset = load_from_disk(DATASET_PTH)

dataset = dataset.map(lambda x: {"lang": ["ne"] * len(x["text"])}, batched=True)
print(dataset)

# Rename audio path column to 'audio' if needed, then cast
if "chunked_audio_filepath" in dataset["train"].column_names:
    dataset = dataset.rename_column("chunked_audio_filepath", "audio")

dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
print("Audio column cast complete.")
print(dataset)

# TRAIN / VAL SPLIT (shuffle train only)
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
    tokenizer=tokenizer,
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

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


@torch.no_grad()
def evaluate_model(model, val_loader):
    """Run generation on the validation set and compute WER/CER."""
    model.eval()
    all_preds, all_labels = [], []

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ne", task="transcribe")

    iterator = tqdm(val_loader, desc="Evaluating") if USE_TQDM else val_loader

    for batch in iterator:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        generated_ids = model.generate(
            input_features=batch["input_features"],
            forced_decoder_ids=forced_decoder_ids,
            max_length=MAX_LABEL_LENGTH,
        )

        labels = batch["labels"].clone()
        labels[labels == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend([normalize_text(s) for s in pred_str])
        all_labels.extend([normalize_text(s) for s in label_str])

    wer = wer_metric.compute(predictions=all_preds, references=all_labels)
    cer = cer_metric.compute(predictions=all_preds, references=all_labels)

    model.train()
    return {"wer": wer, "cer": cer}



# DATALOADERS
train_loader = DataLoader(
    dataset['train'],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    dataset['valid'],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)



# OPTIMISER + SCHEDULER
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)



# TRAINING LOOP (with epoch-level checkpointing + early stopping)
print(f"\nStarting EWC fine-tuning for {NUM_EPOCHS} epoch(s)...")
print(f"  EWC λ          : {EWC_LAMBDA}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Grad accum     : {GRAD_ACCUM_STEPS}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Total steps    : {total_steps}\n")

global_step = 0

best_metric = float("inf")   # lower WER is better
best_epoch  = -1
patience_counter = 0

for epoch in range(6, NUM_EPOCHS + 1):
    model.train()

    epoch_ce_loss  = 0.0
    epoch_ewc_loss = 0.0
    epoch_total    = 0.0

    optimizer.zero_grad()

    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if USE_TQDM else train_loader

    step = 0
    for step, batch in enumerate(iterator, start=1):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # Forward pass
        outputs  = model(
            input_features=batch["input_features"],
            labels=batch["labels"],
        )
        ce_loss  = outputs.loss                  # cross-entropy (ASR loss)
        ewc_loss = ewc_penalty(model)            # EWC regularisation term

        total_loss = ce_loss + EWC_LAMBDA * ewc_loss

        # Scale for gradient accumulation
        (total_loss / GRAD_ACCUM_STEPS).backward()

        # Accumulate stats
        epoch_ce_loss  += ce_loss.item()
        epoch_ewc_loss += ewc_loss.item()
        epoch_total    += total_loss.item()

        # Optimiser step every GRAD_ACCUM_STEPS batches
        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg_ce  = epoch_ce_loss  / step
                avg_ewc = epoch_ewc_loss / step
                avg_tot = epoch_total    / step
                msg = (
                    f"[Epoch {epoch} | Step {global_step}] "
                    f"CE={avg_ce:.4f}  "
                    f"EWC={EWC_LAMBDA * avg_ewc:.4f}  "
                    f"Total={avg_tot:.4f}"
                )
                if USE_TQDM:
                    iterator.set_postfix_str(
                        f"CE={avg_ce:.4f} EWC={EWC_LAMBDA * avg_ewc:.4f}"
                    )
                else:
                    print(msg)

    # Handle leftover batches (if dataset size not divisible by GRAD_ACCUM_STEPS)
    if step % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Epoch summary (training)
    n = len(train_loader)
    print(
        f"\nEpoch {epoch} summary — "
        f"CE: {epoch_ce_loss/n:.4f}  |  "
        f"EWC (scaled): {EWC_LAMBDA * epoch_ewc_loss/n:.4f}  |  "
        f"Total: {epoch_total/n:.4f}"
    )

    # Validation 
    val_metrics = evaluate_model(model, val_loader)
    print(f"Epoch {epoch} validation — WER: {val_metrics['wer']:.4f}  CER: {val_metrics['cer']:.4f}")

    # Epoch-level checkpoint (always saved) 
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}")
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    # Early stopping check 
    current_metric = val_metrics[EARLY_STOPPING_METRIC]

    if current_metric < best_metric:
        best_metric = current_metric
        best_epoch  = epoch
        patience_counter = 0

        # Save best model separately
        best_path = os.path.join(CHECKPOINT_DIR, "best_model")
        model.save_pretrained(best_path)
        processor.save_pretrained(best_path)
        print(f"  New best {EARLY_STOPPING_METRIC.upper()}: {best_metric:.4f} → saved to {best_path}")
    else:
        patience_counter += 1
        print(f"  No improvement ({EARLY_STOPPING_METRIC.upper()}={current_metric:.4f}, "
              f"best={best_metric:.4f}). Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}. "
                  f"Best epoch was {best_epoch} with {EARLY_STOPPING_METRIC.upper()}={best_metric:.4f}.")
            break



# SAVE FINAL MODEL (last completed epoch's weights)

model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nFinal model saved → {OUTPUT_DIR}")
print(f"Best epoch: {best_epoch} (val {EARLY_STOPPING_METRIC.upper()}={best_metric:.4f})")
print("Training complete.")