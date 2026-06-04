# coding=utf-8
"""
AdaMix + Whisper — Training Script
====================================
Loss = CE_loss + lambda_consistency * consistency_loss

Where:
  - CE_loss         : standard cross-entropy on token labels
  - consistency_loss: symmetric KL divergence between two independent
                      stochastic routing passes (A and B), each sampling
                      j ~ U[0,M) for W_up and k ~ U[0,M) for W_down
                      independently, as described in AdaMix paper Section 3.2.

Key design decisions ():
  1.  Two stochastic forward passes per training step — encoder is run once
      and its output is cached; only the decoder is re-run for pass B.
  2.  consistency_loss = 0.5 * [KL(z_A || z_B) + KL(z_B || z_A)]
  3.  Adapters are merged (averaged) before every evaluation loop and
      unmerged afterward so training continues with stochastic routing.
  4.  Final checkpoint saved with adapters merged (inference-ready).
  5.  return_dict=True forced so encoder_last_hidden_state is always
      accessible by name in the consistency-loss second pass.
  6.  use_cache=False during training to be consistent with gradient
      checkpointing; re-enabled during merged evaluation.
  7.  Full processor (not just feature_extractor) passed to Seq2SeqTrainer
      so tokenizer state is preserved in saved checkpoints.
  8.  EarlyStoppingCallback patience=5 — appropriate for adapter-only
      fine-tuning where improvement is gradual.
  9.  compute_metrics handles 3-D logit arrays (argmax before decode).
  10. AdaMix only on the LAST decoder layer (layer_idx == decoder_layers-1).
"""
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import io
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import evaluate

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import BaseModelOutput


from MemoryModule.ADAMIX.last_layer_ada_mix import (
    WhisperForConditionalGeneration as WhisperCustom,
)



# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "openai/whisper-small"
DATA_PATH = "artifacts/Data/indic_voice"
OUTPUT_DIR = "artifacts/transcribe/adamix_adapter/sft/last_layer_"

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

# Consistency loss weight (λ in the paper; 1.0 is the default from AdaMix paper)
LAMBDA_CONSISTENCY = 1.0

print("Using device:", DEVICE)



# LOAD PROCESSOR / TOKENIZER

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

# Force return_dict=True so outputs.encoder_last_hidden_state is always a named
# attribute — required by the consistency-loss second pass which caches the
# encoder output and only re-runs the decoder.
model.config.return_dict = True
model.config.use_cache = False  # Required when gradient_checkpointing=True

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# Set lambda_consistency on the model so forward() picks it up automatically
model.lambda_consistency = LAMBDA_CONSISTENCY

# Generation config — Nepali transcription
model.generation_config.language = "ne"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.use_cache = False   # consistent with model.config

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []



# LOAD DATASET

dataset = load_from_disk(DATA_PATH)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.shuffle(seed=SEED)
dataset["train"] = dataset["train"].shuffle(seed=SEED)

print("Audio column cast complete.")
print(dataset)
print("Train raw size:", len(dataset["train"]))
print("Val raw size  :", len(dataset["valid"]))



# DATA COLLATOR

@dataclass
class DataCollatorSpeechSeq2SeqTranscriptionOnly:
    processor: Any
    tokenizer: Any
    decoder_start_token_id: int
    max_label_length: int = 448

    def _decode_audio(self, audio_obj, idx: int):
        # Case 1: torchcodec AudioDecoder
        if hasattr(audio_obj, "get_all_samples"):
            try:
                samples = audio_obj.get_all_samples()
                audio_array = samples.data
                sr = samples.sample_rate
                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()
                if audio_array.ndim == 2:
                    audio_array = (
                        audio_array.mean(axis=0)
                        if audio_array.shape[0] <= 8
                        else audio_array.mean(axis=1)
                    )
                return audio_array.astype("float32"), sr
            except Exception as e:
                raise ValueError(
                    f"Failed to decode AudioDecoder at batch index {idx}: {e}"
                )

        # Case 2: HF dict with array
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            audio_array = audio_obj["array"]
            sr = audio_obj.get("sampling_rate", 16000)
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)
            return audio_array.astype("float32"), sr

        # Case 3: dict with raw bytes
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
                raise ValueError(
                    f"Failed to decode audio bytes at batch index {idx}: {e}"
                )

        raise ValueError(
            f"Unsupported audio type at batch index {idx}: {type(audio_obj)}"
        )

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
                print(f"\nSkipping bad audio sample — index {i}, error: {e}")
                continue

        if not audio_arrays:
            raise ValueError("All audio samples in this batch failed to decode.")

        batch_inputs = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        )

        labels = [
            {
                "input_ids": self.tokenizer(
                    f["text"],
                    truncation=True,
                    max_length=self.max_label_length,
                    return_tensors=None,
                )["input_ids"]
            }
            for f in valid_features
        ]

        labels_batch = self.tokenizer.pad(labels, padding=True, return_tensors="pt")
        labels_tensor = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # Strip leading decoder_start_token if the tokenizer prepended it
        if (
            labels_tensor.size(1) > 0
            and (labels_tensor[:, 0] == self.decoder_start_token_id).all()
        ):
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

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()

    # Guard: if generate returned logits (3-D tensor) instead of token IDs, take argmax
    if pred_ids.ndim == 3:
        pred_ids = pred_ids.argmax(axis=-1)

    label_ids[label_ids == -100] = tokenizer_transcribe.pad_token_id

    pred_str  = tokenizer_transcribe.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer_transcribe.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [normalize_text(s) for s in pred_str]
    label_str = [normalize_text(s) for s in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}



# FREEZE / UNFREEZE
# Only AdaMix parameters (W_up and W_down experts) are trainable.
# All other Whisper parameters remain frozen.

for p in model.parameters():
    p.requires_grad = False

TRAINABLE_KEYWORDS = ["ada_mix"]

for name, param in model.named_parameters():
    if any(key in name.lower() for key in TRAINABLE_KEYWORDS):
        param.requires_grad = True
        print(f"Trainable: {name}")

num_total     = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal params     : {num_total:,}")
print(f"Trainable params : {num_trainable:,}")
print(f"Trainable %      : {100 * num_trainable / num_total:.4f}%")

if num_trainable == 0:
    raise RuntimeError(
        "No trainable parameters found. "
        "Check that AdaMix layers exist and TRAINABLE_KEYWORDS matches their names."
    )



# CONSISTENCY LOSS HELPER
def adamix_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric KL divergence between two stochastic routing passes.

    Formula (paper Section 3.2, Equation 5):
        L_consistency = 0.5 * [KL(z_A || z_B) + KL(z_B || z_A)]

    Args:
        logits_a: (batch, seq_len, vocab)  from routing pass A
        logits_b: (batch, seq_len, vocab)  from routing pass B

    Returns:
        scalar consistency loss
    """
    log_p_a = F.log_softmax(logits_a, dim=-1)
    log_p_b = F.log_softmax(logits_b, dim=-1)
    p_a = log_p_a.exp()
    p_b = log_p_b.exp()

    # kl_div(input=log_Q, target=P) computes KL(P || Q)
    kl_ab = F.kl_div(log_p_b, p_a, reduction="batchmean")   # KL(A || B)
    kl_ba = F.kl_div(log_p_a, p_b, reduction="batchmean")   # KL(B || A)

    return 0.5 * (kl_ab + kl_ba)



# ADAMIX TRAINER
# Overrides:
#   1. compute_loss  — adds consistency regularisation to CE loss
#   2. evaluation_loop — merges adapters before eval, unmerges after
#   3. save_model    — state_dict sanity check

class AdaMixTrainer(Seq2SeqTrainer):
    

    # Total loss = CE + λ * consistency
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
     
        labels = inputs.get("labels")

        
        outputs_a = model(
            input_features=inputs["input_features"],
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        ce_loss    = outputs_a.loss        # cross-entropy from pass A
        lm_logits_a = outputs_a.logits    # (batch, seq_len, vocab)

        # Only add consistency loss during training when adapters are present
        has_adapters = any(
            getattr(layer, "use_adapter", False)
            for layer in model.model.decoder.layers
        )
        lambda_c = getattr(model, "lambda_consistency", LAMBDA_CONSISTENCY)

        if model.training and has_adapters and lambda_c > 0.0:
        
            cached_encoder_output = BaseModelOutput(
                last_hidden_state=outputs_a.encoder_last_hidden_state.detach()
            )

            # Build decoder_input_ids the same way the model does internally
            from transformers.models.whisper.modeling_whisper import shift_tokens_right
            decoder_input_ids = shift_tokens_right(
                labels,
                model.config.pad_token_id,
                model.config.decoder_start_token_id,
            )

           
            outputs_b = model(
                input_features=None,            # encoder skipped — reuse cache
                encoder_outputs=cached_encoder_output,
                decoder_input_ids=decoder_input_ids,
                labels=None,                    # no CE for pass B
                use_cache=False,
                return_dict=True,
            )
            lm_logits_b = outputs_b.logits     

            # Consistency loss (symmetric KL) 
            c_loss = adamix_consistency_loss(lm_logits_a, lm_logits_b)

            total_loss = ce_loss + lambda_c * c_loss

            # Log both components so they appear in TensorBoard / W&B
            # if self.state.global_step % self.args.logging_steps == 0:
            #     self.log({
            #         "ce_loss":          ce_loss.item(),
            #         "consistency_loss": c_loss.item(),
            #         "total_loss":       total_loss.item(),
            #     })
        else:
            # Evaluation or no adapters — CE loss only
            total_loss = ce_loss

        return (total_loss, outputs_a) if return_outputs else total_loss

    def evaluation_loop(self, *args, **kwargs):
       
        self.model.merge_adapters()
        # Re-enable use_cache for generation (merged = deterministic = safe)
        self.model.config.use_cache = True
        self.model.generation_config.use_cache = True

        torch.cuda.empty_cache()
        result = super().evaluation_loop(*args, **kwargs)
        torch.cuda.empty_cache()

        # Restore stochastic training mode
        self.model.unmerge_adapters()
        self.model.config.use_cache = False
        self.model.generation_config.use_cache = False

        return result

    #  Checkpoint sanity check
    def save_model(self, output_dir=None, _internal_call=False):
        state_dict = self.model.state_dict()
        for name, _ in self.model.named_parameters():
            if name not in state_dict:
                print(f"WARNING — Missing key in state_dict: {name}")
        super().save_model(output_dir, _internal_call)



# TRAINING ARGUMENTS
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
    greater_is_better=False,        # lower WER is better
    save_total_limit=2,
    report_to=["tensorboard"],
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)





trainer = AdaMixTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,            
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,          
            early_stopping_threshold=1e-5,
        )
    ],
)



# TRAIN
result = trainer.evaluate()
print(f"\nInitial evaluation results: {result}")
trainer.train(resume_from_checkpoint=True)



# SAVE BEST MODEL (inference-ready: adapters merged)
print("\nMerging AdaMix adapters for inference-ready checkpoint …")
model.merge_adapters()

best_dir = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(best_dir)
processor.save_pretrained(best_dir)
tokenizer_transcribe.save_pretrained(
    os.path.join(best_dir, "tokenizer_transcribe")
)

print(f"\nDone. Best model saved to: {best_dir}")