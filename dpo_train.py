import os
import re
import io
import random
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import evaluate

from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    WhisperForConditionalGeneration
)

from MemoryModule.conponents.TitansLastLayerModelingNew import (
    WhisperForConditionalGeneration as WhisperCustom,
)

from dpo_trainer import WhisperDPOTrainer


# SEED / DEVICE


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)



# PATHS

BASE_MODEL = "openai/whisper-small"

# SFT_CHECKPOINT = (
#     "artifacts/checkpoint/n_gram_attention_memory/"
#     "last_layer/sft/gram_one_memory_only/best_model"
# )

# REF_CHECKPOINT = SFT_CHECKPOINT

DATA_PATH = "artifacts/Data/indic_voice_dpo"

OUTPUT_DIR = (
   "artifacts/transcribe/memory/memory_3_last_layer/dpo"
)

PRECOMPUTE_CACHE_DIR = os.path.join(OUTPUT_DIR, "ref_logps_cache")



# HYPERPARAMETERS

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 16

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 30

MAX_LABEL_LENGTH = 448
GEN_MAX_LENGTH = 256

BETA = 0.5

# First keep False.
# After everything works, you can set True and use the precompute block.
PRECOMPUTE_REF_LOGPS = True



# NORMALIZATION

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


# PROCESSOR / TOKENIZER


processor = WhisperProcessor.from_pretrained(BASE_MODEL)

tokenizer_transcribe = WhisperTokenizer.from_pretrained(
    BASE_MODEL,
    language="ne",
    task="transcribe",
)

processor.tokenizer = tokenizer_transcribe


# POLICY MODEL


torch.manual_seed(SEED)

model = WhisperCustom.from_pretrained(BASE_MODEL).to(DEVICE)

# Important:
# Do NOT overwrite proj_out.weight after loading SFT checkpoint.
# Your SFT checkpoint already contains trained weights.

model.config.use_cache = False

model.generation_config.language = "ne"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Gradient checkpointing can break when most parameters are frozen.
# enable_input_require_grads helps gradients flow through checkpointed blocks.
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

# print("\nPolicy model loaded from:", SFT_CHECKPOINT)


# FREEZE / UNFREEZE


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
    "router_proj",
    "mem_block"
]

for name, param in model.named_parameters():
    if any(k.lower() in name.lower() for k in TRAINABLE_KEYWORDS):
        param.requires_grad = True

print("\nTrainable parameters:")
num_total = 0
num_trainable = 0

for name, param in model.named_parameters():
    num_total += param.numel()

    if param.requires_grad:
        num_trainable += param.numel()
        print(f"  {name}")

print(f"\nTotal params     : {num_total:,}")
print(f"Trainable params : {num_trainable:,}")

if num_trainable == 0:
    raise ValueError(
        "No trainable parameters found. "
        "Check TRAINABLE_KEYWORDS and your custom module names."
    )

print("\nRouter proj values after loading:")
for name, param in model.named_parameters():
    if "router_proj" in name.lower():
        print(f"  {name}: first 5 values = {param.data.flatten()[:5].tolist()}")



# REFERENCE MODEL

ref_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
ref_model.eval()

ref_model.config.use_cache = False

ref_model.generation_config.language = "ne"
ref_model.generation_config.task = "transcribe"
ref_model.generation_config.forced_decoder_ids = None
ref_model.generation_config.suppress_tokens = []

ref_model.config.forced_decoder_ids = None
ref_model.config.suppress_tokens = []

if hasattr(ref_model, "gradient_checkpointing_disable"):
    ref_model.gradient_checkpointing_disable()

for p in ref_model.parameters():
    p.requires_grad = False

# print("\nReference model loaded from same custom architecture:", REF_CHECKPOINT)




dataset = load_from_disk(DATA_PATH)
print(dataset)

dataset = dataset.filter(
    lambda x: (
        normalize_text(x.get("text", ""))
        != normalize_text(x.get("rejected", ""))
    )
    and normalize_text(x.get("text", "")) != ""
    and normalize_text(x.get("rejected", "")) != "",
    desc="Removing invalid or identical chosen/rejected pairs",
)

dataset = dataset.shuffle(seed=SEED)

print("Train size:", len(dataset["train"]))
print("Val size  :", len(dataset["valid"]))




@dataclass
class DataCollatorSpeechSeq2SeqDPOTranscribe:
    processor: Any
    tokenizer: Any
    decoder_start_token_id: int
    max_label_length: int = 448

    def _decode_audio(self, audio_obj, idx: int) -> Tuple[np.ndarray, int]:
        """
        Decode audio robustly.
        Expected output:
            audio_array: float32 numpy array
            sr: sampling rate
        """

        # Case 1: HuggingFace / torchcodec AudioDecoder
        if hasattr(audio_obj, "get_all_samples"):
            try:
                samples = audio_obj.get_all_samples()
                audio_array = samples.data
                sr = samples.sample_rate

                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()

                audio_array = np.asarray(audio_array, dtype=np.float32)

                if audio_array.ndim == 2:
                    if audio_array.shape[0] <= 8:
                        audio_array = audio_array.mean(axis=0)
                    else:
                        audio_array = audio_array.mean(axis=1)

                audio_array = audio_array.squeeze()

                if audio_array.size == 0:
                    raise ValueError("empty audio array")

                return audio_array.astype("float32"), sr

            except Exception as e:
                raise ValueError(f"AudioDecoder failed at index {idx}: {e}")

        # Case 2: dict with array
        if isinstance(audio_obj, dict) and "array" in audio_obj:
            audio_array = audio_obj["array"]
            sr = audio_obj.get("sampling_rate", 16000)

            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            audio_array = np.asarray(audio_array, dtype=np.float32)

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)

            audio_array = audio_array.squeeze()

            if audio_array.size == 0:
                raise ValueError("empty audio array")

            return audio_array.astype("float32"), sr

        # Case 3: dict with bytes
        if isinstance(audio_obj, dict) and "bytes" in audio_obj:
            try:
                buffer = io.BytesIO(audio_obj["bytes"])
                audio_array, sr = sf.read(buffer)

                if isinstance(audio_array, torch.Tensor):
                    audio_array = audio_array.cpu().numpy()

                audio_array = np.asarray(audio_array, dtype=np.float32)

                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=-1)

                audio_array = audio_array.squeeze()

                if audio_array.size == 0:
                    raise ValueError("empty audio array")

                return audio_array.astype("float32"), sr

            except Exception as e:
                raise ValueError(f"Audio bytes failed at index {idx}: {e}")

        raise ValueError(
            f"Unsupported audio type at batch index {idx}: {type(audio_obj)}"
        )

    def _tokenize_labels(self, texts: List[str]) -> torch.Tensor:
        labels = []

        for text in texts:
            text = normalize_text(text)

            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_label_length,
                return_tensors=None,
            )

            labels.append({"input_ids": tokenized["input_ids"]})

        batch = self.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt",
        )

        labels_tensor = batch["input_ids"].masked_fill(
            batch["attention_mask"].ne(1),
            -100,
        )

        # Remove decoder start token if tokenizer already inserted it.
        if labels_tensor.size(1) > 0:
            first_col = labels_tensor[:, 0]
            if (first_col == self.decoder_start_token_id).all().item():
                labels_tensor = labels_tensor[:, 1:]

        return labels_tensor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = []
        valid_features = []

        for i, f in enumerate(features):
            if "audio" not in f:
                print(f"[SKIP] Missing audio at batch index {i}")
                continue

            try:
                audio_array, sr = self._decode_audio(f["audio"], i)

                if sr != 16000:
                    print(f"[SKIP] Expected 16000 Hz, got {sr} at batch index {i}")
                    continue

                chosen = normalize_text(f.get("text", ""))
                rejected = normalize_text(f.get("rejected", ""))

                if chosen == "" or rejected == "" or chosen == rejected:
                    continue

                audio_arrays.append(audio_array)
                valid_features.append(f)

            except Exception as e:
                print(f"[SKIP BAD AUDIO] batch_index={i}, error={e}")
                continue

        if len(audio_arrays) == 0:
            raise ValueError(
                "All samples in this batch failed or had invalid DPO labels."
            )

        batch_inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        )

        chosen_texts = [f["text"] for f in valid_features]
        rejected_texts = [f["rejected"] for f in valid_features]

        chosen_labels_tensor = self._tokenize_labels(chosen_texts)
        rejected_labels_tensor = self._tokenize_labels(rejected_texts)

        batch = {
            "input_features": batch_inputs["input_features"],
            "chosen_labels": chosen_labels_tensor,
            "rejected_labels": rejected_labels_tensor,
        }

        # Pass precomputed reference log-probs if present.
        if valid_features and "ref_chosen_logps" in valid_features[0]:
            batch["ref_chosen_logps"] = torch.tensor(
                [f["ref_chosen_logps"] for f in valid_features],
                dtype=torch.float32,
            )

        if valid_features and "ref_rejected_logps" in valid_features[0]:
            batch["ref_rejected_logps"] = torch.tensor(
                [f["ref_rejected_logps"] for f in valid_features],
                dtype=torch.float32,
            )

        return batch


data_collator = DataCollatorSpeechSeq2SeqDPOTranscribe(
    processor=processor,
    tokenizer=tokenizer_transcribe,
    decoder_start_token_id=model.config.decoder_start_token_id,
    max_label_length=MAX_LABEL_LENGTH,
)



print("\nRunning collator sanity check...")

try:
    n_check = min(4, len(dataset["train"]))
    debug_features = [dataset["train"][i] for i in range(n_check)]
    debug_batch = data_collator(debug_features)

    print("Collator output keys:", debug_batch.keys())
    print("input_features:", debug_batch["input_features"].shape)
    print("chosen_labels:", debug_batch["chosen_labels"].shape)
    print("rejected_labels:", debug_batch["rejected_labels"].shape)

except Exception as e:
    print("[WARNING] Collator sanity check failed:", e)
    print("Training may still work if the first few samples were bad.")



# METRICS
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.detach().cpu().numpy()

    if isinstance(label_ids, torch.Tensor):
        label_ids = label_ids.detach().cpu().numpy()

    label_ids = np.where(
        label_ids == -100,
        tokenizer_transcribe.pad_token_id,
        label_ids,
    )

    pred_ids = np.where(
        pred_ids == -100,
        tokenizer_transcribe.pad_token_id,
        pred_ids,
    )

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

    pairs = [
        (p, l)
        for p, l in zip(pred_str, label_str)
        if len(l.strip()) > 0
    ]

    if not pairs:
        return {
            "wer": 1.0,
            "cer": 1.0,
        }

    filtered_preds, filtered_labels = zip(*pairs)

    return {
        "wer": wer_metric.compute(
            predictions=list(filtered_preds),
            references=list(filtered_labels),
        ),
        "cer": cer_metric.compute(
            predictions=list(filtered_preds),
            references=list(filtered_labels),
        ),
    }


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

    eval_strategy="steps",
    eval_steps=1000,

    logging_strategy="steps",
    logging_steps=200,

    save_strategy="steps",
    save_steps=1000,

    max_grad_norm=0.5,

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

    # Keep 0 first for debugging audio/collator issues.
    # Later you can increase to 2 or 4.
    dataloader_num_workers=0,
    dataloader_pin_memory=True,

    seed=SEED,
    data_seed=SEED,
)


# TRAINER CLASS
class DebugWhisperDPOTrainer(WhisperDPOTrainer):
    def evaluation_loop(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = super().evaluation_loop(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result



# TRAINER
trainer = DebugWhisperDPOTrainer(
    model=model,
    ref_model=ref_model,
    processor=processor,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],

    data_collator=data_collator,
    compute_metrics=compute_metrics,

    beta=0.1,
    precompute_ref_logps=True,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.00001,
        )
    ],

    loss_types=["sigmoid", "sft"],
    loss_weights=[1.0, 0.2],

    length_normalize_logps=True,
    label_smoothing=0.0,
    label_pad_token_id=-100,

    tokenizer=processor.feature_extractor,
)



print("\nRunning initial evaluation on SFT checkpoint...")
results = trainer.evaluate()

print("\nBaseline Evaluation Results before DPO training:")
print(results)

baseline_wer = results.get("eval_wer", None)

print(f"\nBaseline WER: {baseline_wer}")
print("─" * 60)
print("If final WER > baseline WER, DPO caused forgetting.")
print("─" * 60)


if PRECOMPUTE_REF_LOGPS:
    os.makedirs(PRECOMPUTE_CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(PRECOMPUTE_CACHE_DIR, "train")
    valid_cache = os.path.join(PRECOMPUTE_CACHE_DIR, "valid")

    if os.path.exists(train_cache) and os.path.exists(valid_cache):
        print("Loading cached precomputed reference log-probs...")
        train_dataset = load_from_disk(train_cache)
        eval_dataset = load_from_disk(valid_cache)
    else:
        print("Precomputing reference log-probs for train set...")
        train_dataset = trainer._precompute_ref_logps(
            dataset["train"],
            batch_size=EVAL_BATCH_SIZE,
            desc="Precomputing train ref logps",
            normalize_fn=normalize_text,
        )

        print("Precomputing reference log-probs for valid set...")
        eval_dataset = trainer._precompute_ref_logps(
            dataset["valid"],
            batch_size=EVAL_BATCH_SIZE,
            desc="Precomputing valid ref logps",
            normalize_fn=normalize_text,
        )

        train_dataset.save_to_disk(train_cache)
        eval_dataset.save_to_disk(valid_cache)

    trainer.train_dataset = train_dataset
    trainer.eval_dataset = eval_dataset
    trainer.precompute_ref_logps = True

    # Optional: free reference model from GPU after precompute
    trainer.ref_model = None
    del ref_model
    torch.cuda.empty_cache()




print("\nStarting DPO training...")
trainer.train()




print("\nRunning final evaluation...")
final_results = trainer.evaluate()

print("\nFinal Evaluation Results after DPO training:")
print(final_results)

final_wer = final_results.get("eval_wer", None)

print("\n" + "═" * 60)
print(f"Baseline WER SFT : {baseline_wer}")
print(f"Final WER DPO    : {final_wer}")

if baseline_wer is not None and final_wer is not None:
    delta = final_wer - baseline_wer

    if delta < 0:
        print(f"WER change       : {delta:.4f} DPO improved WER")
    elif delta == 0:
        print(f"WER change       : {delta:.4f} No change")
    else:
        print(f"WER change       : +{delta:.4f} DPO caused forgetting")

print("═" * 60)



best_dir = os.path.join(OUTPUT_DIR, "best_model")

trainer.save_model(best_dir)
processor.save_pretrained(best_dir)

tokenizer_transcribe.save_pretrained(
    os.path.join(best_dir, "tokenizer_transcribe")
)

print(f"\nSaved best model to: {best_dir}")