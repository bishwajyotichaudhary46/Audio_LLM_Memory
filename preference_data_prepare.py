#!/usr/bin/env python3
# coding=utf-8

"""
Full fixed DPO rejected-generation script for your custom Whisper + Titan memory model.

Features:
1. Verifies one sample first:
      CHOSEN   = ground-truth transcription
      REJECTED = model-generated transcription
2. Generates rejected text for full dataset.
3. Uses JSONL cache during generation.
4. If interrupted, rerun the script and it resumes from cache.
5. Saves final dataset only after all cache rows are complete.
6. Removes cache only after successful save_to_disk().
7. Fixes your previous error:
      [PIPE BATCH ERROR] tuple index out of range
      KeyError: 'raw'

Important:
This script does NOT use Hugging Face pipeline batching for raw audio.
Pipeline batching caused your tuple-index error.
Instead, it uses:
      raw arrays -> WhisperFeatureExtractor -> input_features -> model.generate -> tokenizer.batch_decode
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from MemoryModule.conponents.TitansLastLayerModelingNew import (
    WhisperForConditionalGeneration as WhisperCustom
)



# CONFIG


DATASET_PATH = "artifacts/Data/indic_voice"
SAVE_PATH = "artifacts/Data/indic_voice_dpo_memory_gating"

MODEL_PATH = "artifacts/checkpoint/memory/sft/memory_layer_3_gate/best_model"

CACHE_DIR = "artifacts/cache/rejected_generation_cache"

SAMPLING_RATE = 16000
MAX_AUDIO_SEC = 30


DTYPE = torch.float32

GEN_BATCH_SIZE = 4

MAX_NEW_TOKENS = 128
NUM_BEAMS = 1
DO_SAMPLE = False

VERIFY_ONLY = False
STOP_IF_VERIFY_EMPTY = True

FILTER_BAD_PAIRS = True
SKIP_IDENTICAL_CHOSEN_REJECTED = True


DISABLE_MEMORY_WRITE_IF_AVAILABLE = False

CACHE_FLUSH_EVERY = 20

SPLITS_TO_PROCESS = ["train", "valid", "validation", "test"]

CHOSEN_COLUMN_CANDIDATES = [
    "transcription",
    "sentence",
    "text",
    "normalized_text",
    "chosen",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# BASIC UTILS
def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def get_chosen(example: Dict[str, Any]) -> str:
    for col in CHOSEN_COLUMN_CANDIDATES:
        if col in example and example[col] is not None:
            value = str(example[col]).strip()
            if value:
                return value
    return ""


def make_keep_flag(chosen: str, rejected: str) -> bool:
    chosen_norm = normalize_text(chosen)
    rejected_norm = normalize_text(rejected)

    if not chosen_norm:
        return False

    if not rejected_norm:
        return False

    if SKIP_IDENTICAL_CHOSEN_REJECTED and chosen_norm == rejected_norm:
        return False

    return True


def set_memory_write_flag(model: torch.nn.Module, enabled: bool) -> None:
    """
    Optional helper. This only works if your custom modules check these attributes.
    It will not break if the attributes are unused.
    """
    for module in model.modules():
        if module.__class__.__name__ in [
            "NeuralLongTermMemory",
            "LMMBlock",
            "WhisperMemoryLayer",
        ]:
            try:
                module.disable_memory_write = not enabled
            except Exception:
                pass
            try:
                module.memory_write_enabled = enabled
            except Exception:
                pass



# AUDIO EXTRACTION
def extract_audio_array(item: Any) -> Optional[np.ndarray]:
    """
    Handles:
      1. HF audio dict: {"array": ..., "sampling_rate": ...}
      2. dict with bytes/path
      3. torchcodec AudioDecoder with get_all_samples()
      4. raw torch.Tensor
      5. raw np.ndarray
    """
    arr = None
    sr = SAMPLING_RATE

    try:
        if isinstance(item, dict):
            if "array" in item and item["array"] is not None:
                arr = np.asarray(item["array"], dtype=np.float32)
                sr = int(item.get("sampling_rate", SAMPLING_RATE))

            elif "raw" in item and item["raw"] is not None:
                arr = np.asarray(item["raw"], dtype=np.float32)
                sr = int(item.get("sampling_rate", SAMPLING_RATE))

            elif "bytes" in item or "path" in item:
                import io
                import soundfile as sf

                if item.get("bytes") is not None:
                    raw = item["bytes"]
                elif item.get("path") is not None:
                    with open(item["path"], "rb") as f:
                        raw = f.read()
                else:
                    return None

                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                arr = np.asarray(arr, dtype=np.float32)

            else:
                print(f"[SKIP] Unknown audio dict keys: {list(item.keys())}")
                return None

        elif hasattr(item, "get_all_samples"):
            # datasets.features._torchcodec.AudioDecoder
            frames = item.get_all_samples()
            arr = frames.data

            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()

            arr = np.asarray(arr, dtype=np.float32)

            if hasattr(frames, "sample_rate"):
                sr = int(frames.sample_rate)
            elif hasattr(item, "sample_rate"):
                sr = int(item.sample_rate)

        elif isinstance(item, torch.Tensor):
            arr = item.detach().cpu().numpy().astype(np.float32)

        elif isinstance(item, np.ndarray):
            arr = item.astype(np.float32)

        else:
            print(f"[SKIP] Unknown audio type: {type(item)}")
            return None

        if arr is None:
            return None

        # Stereo / multi-channel to mono.
        if arr.ndim == 2:
            # [channels, time]
            if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                arr = arr.mean(axis=0)
            # [time, channels]
            else:
                arr = arr.mean(axis=1)

        arr = np.squeeze(arr).astype(np.float32)

        if arr.size == 0:
            return None

        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if sr != SAMPLING_RATE:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLING_RATE)
            arr = arr.astype(np.float32)

        max_len = SAMPLING_RATE * MAX_AUDIO_SEC
        if arr.shape[0] > max_len:
            arr = arr[:max_len]

        return arr

    except Exception as e:
        print(f"[BAD AUDIO] {e}")
        traceback.print_exc()
        return None



# MODEL LOAD
def load_model_and_processor():
    print("=" * 80)
    print("Loading model and processor")
    print("=" * 80)
    print(f"Model path : {MODEL_PATH}")
    print(f"Device     : {DEVICE}")
    print(f"DType      : {DTYPE}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    # Same task style as your previous working code.
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small",
        task="translate",
    )

    model = WhisperCustom.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # Safer for custom modified decoder.
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model.generation_config.forced_decoder_ids = None

    if DISABLE_MEMORY_WRITE_IF_AVAILABLE:
        set_memory_write_flag(model, enabled=False)

    print("Model and processor ready.")
    print("=" * 80)

    return model, feature_extractor, tokenizer



# GENERATION
def generate_batch_direct(
    model,
    feature_extractor,
    tokenizer,
    arrays: List[np.ndarray],
) -> List[str]:
    """
    Direct batched generation.

    This avoids:
        asr_pipe(audio_inputs, batch_size=...)
    because pipeline batching produced:
        IndexError: tuple index out of range
    in your previous run.

    Input:
        arrays = list of 1-D float32 audio arrays

    Output:
        list of decoded strings
    """
    if not arrays:
        return []

    valid_positions = []
    valid_arrays = []

    for i, arr in enumerate(arrays):
        if arr is None:
            continue

        arr = np.asarray(arr, dtype=np.float32)

        if arr.size == 0:
            continue

        valid_positions.append(i)
        valid_arrays.append(arr)

    outputs = [""] * len(arrays)

    if not valid_arrays:
        return outputs

    try:
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        features = feature_extractor(
            valid_arrays,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=SAMPLING_RATE * MAX_AUDIO_SEC,
            truncation=True,
            return_attention_mask=True,
        )

        input_features = features["input_features"].to(
            device=model_device,
            dtype=model_dtype,
        )

        attention_mask = None
        if "attention_mask" in features:
            attention_mask = features["attention_mask"].to(model_device)

        model.config.use_cache = False
        model.generation_config.use_cache = False
        model.generation_config.forced_decoder_ids = None

        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_beams": NUM_BEAMS,
            "do_sample": DO_SAMPLE,
            "use_cache": False,
        }

 
        # torch.autograd.grad() internally during generation.
        if attention_mask is not None:
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        else:
            generated_ids = model.generate(
                input_features=input_features,
                **gen_kwargs,
            )

        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for pos, text in zip(valid_positions, decoded):
            outputs[pos] = str(text).strip()

        clean_cuda()
        return outputs

    except Exception as e:
        print(f"[DIRECT BATCH ERROR] {e}")
        traceback.print_exc()
        clean_cuda()

        # Safe fallback: direct one-by-one generation.
        print("[INFO] Falling back to direct one-by-one generation.")

        for pos, arr in zip(valid_positions, valid_arrays):
            try:
                one_output = generate_batch_direct(
                    model=model,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    arrays=[arr],
                )
                outputs[pos] = one_output[0] if one_output else ""
            except Exception as inner_e:
                print(f"[DIRECT SINGLE ERROR] index={pos}: {inner_e}")
                traceback.print_exc()
                outputs[pos] = ""
                clean_cuda()

        return outputs


def generate_one_direct(model, feature_extractor, tokenizer, arr: np.ndarray) -> str:
    outputs = generate_batch_direct(
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        arrays=[arr],
    )
    return outputs[0] if outputs else ""


# ONE SAMPLE VERIF
def verify_one_sample(
    dataset,
    model,
    feature_extractor,
    tokenizer,
    split_name: Optional[str] = None,
    index: int = 0,
) -> bool:
    if split_name is None:
        for s in ["train", "valid", "validation", "test"]:
            if s in dataset and len(dataset[s]) > index:
                split_name = s
                break

    if split_name is None:
        raise ValueError("No valid split found for verification.")

    sample = dataset[split_name][index]
    chosen = get_chosen(sample)
    arr = extract_audio_array(sample["audio"])

    print("\n" + "=" * 80)
    print("ONE SAMPLE VERIFICATION")
    print("=" * 80)
    print(f"Split       : {split_name}")
    print(f"Index       : {index}")
    print(f"Audio type  : {type(sample['audio'])}")

    if arr is None:
        print("[FAIL] Audio extraction failed.")
        print("=" * 80 + "\n")
        return False

    print(f"Audio shape : {arr.shape}")
    print(f"Audio dtype : {arr.dtype}")
    print(f"Audio minmax: {arr.min():.5f}, {arr.max():.5f}")

    rejected = generate_one_direct(model, feature_extractor, tokenizer, arr)
    keep = make_keep_flag(chosen, rejected)

    print("-" * 80)
    print(f"CHOSEN   : {chosen}")
    print(f"REJECTED : {rejected}")
    print(f"DPO KEEP : {keep}")
    print("-" * 80)

    ok = bool(rejected.strip())

    if ok:
        print("[PASS] One-sample generation works.")
    else:
        print("[FAIL] Rejected text is empty.")

    print("=" * 80 + "\n")
    return ok



# CACHE
def cache_file_for_split(split: str) -> Path:
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}_generated.jsonl"


def load_split_cache(split: str) -> Dict[int, Dict[str, Any]]:
    path = cache_file_for_split(split)
    cache: Dict[int, Dict[str, Any]] = {}

    if not path.exists():
        return cache

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                idx = int(row["idx"])
                cache[idx] = row
            except Exception:
                continue

    print(f"[CACHE] Loaded {len(cache)} rows from {path}")
    return cache


def append_rows_to_cache(split: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path = cache_file_for_split(split)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[CACHE] Wrote {len(rows)} rows -> {path}")


def remove_cache_dir() -> None:
    path = Path(CACHE_DIR)

    if path.exists():
        shutil.rmtree(path)
        print(f"[CACHE] Removed cache directory: {CACHE_DIR}")



# SPLIT PROCESSING
def generate_split_with_cache(
    dataset_split,
    split_name: str,
    model,
    feature_extractor,
    tokenizer,
):
    total = len(dataset_split)
    cache = load_split_cache(split_name)

    print("\n" + "=" * 80)
    print(f"Processing split: {split_name}")
    print(f"Total rows       : {total}")
    print(f"Already cached   : {len(cache)}")
    print("=" * 80)

    pending_rows: List[Dict[str, Any]] = []

    i = 0

    while i < total:
        if i in cache:
            i += 1
            continue

        batch_indices = []
        batch_chosen = []
        batch_arrays = []

        while i < total and len(batch_arrays) < GEN_BATCH_SIZE:
            if i in cache:
                i += 1
                continue

            ex = dataset_split[i]
            chosen = get_chosen(ex)
            arr = extract_audio_array(ex["audio"])

            if not chosen or arr is None:
                row = {
                    "idx": i,
                    "chosen": chosen,
                    "rejected": "",
                    "dpo_keep": False,
                    "error": "missing_chosen_or_bad_audio",
                }

                cache[i] = row
                pending_rows.append(row)
                i += 1
                continue

            batch_indices.append(i)
            batch_chosen.append(chosen)
            batch_arrays.append(arr)
            i += 1

        if batch_arrays:
            rejected_values = generate_batch_direct(
                model=model,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                arrays=batch_arrays,
            )

            for idx, chosen, rejected in zip(batch_indices, batch_chosen, rejected_values):
                keep = make_keep_flag(chosen, rejected)

                row = {
                    "idx": idx,
                    "chosen": chosen,
                    "rejected": rejected,
                    "dpo_keep": keep,
                    "error": "",
                }

                cache[idx] = row
                pending_rows.append(row)

        if len(pending_rows) >= CACHE_FLUSH_EVERY:
            append_rows_to_cache(split_name, pending_rows)
            pending_rows = []
            clean_cuda()

        print(f"[{split_name}] cached {len(cache)}/{total}", end="\r")

    append_rows_to_cache(split_name, pending_rows)

    print(f"\n[{split_name}] generation cache complete: {len(cache)}/{total}")

    chosen_col = []
    rejected_col = []
    keep_col = []

    missing = []

    for idx in range(total):
        row = cache.get(idx)

        if row is None:
            missing.append(idx)
            chosen_col.append("")
            rejected_col.append("")
            keep_col.append(False)
        else:
            chosen_col.append(str(row.get("chosen", "")))
            rejected_col.append(str(row.get("rejected", "")))
            keep_col.append(bool(row.get("dpo_keep", False)))

    if missing:
        raise RuntimeError(
            f"Cache incomplete for split {split_name}. "
            f"Missing first indices: {missing[:20]}"
        )

    # Replace columns if already present.
    for col in ["chosen", "rejected", "dpo_keep"]:
        if col in dataset_split.column_names:
            dataset_split = dataset_split.remove_columns([col])

    dataset_split = dataset_split.add_column("chosen", chosen_col)
    dataset_split = dataset_split.add_column("rejected", rejected_col)
    dataset_split = dataset_split.add_column("dpo_keep", keep_col)

    if FILTER_BAD_PAIRS:
        before = len(dataset_split)

        dataset_split = dataset_split.filter(
            lambda ex: bool(ex["dpo_keep"]),
            desc=f"Filtering valid DPO pairs for {split_name}",
        )

        after = len(dataset_split)
        print(f"[{split_name}] filtered: {before} -> {after}")

    return dataset_split



# MAIN
def main():
    print("=" * 80)
    print("FULL FIXED DPO REJECTED GENERATION")
    print("=" * 80)
    print(f"Dataset path : {DATASET_PATH}")
    print(f"Model path   : {MODEL_PATH}")
    print(f"Cache dir    : {CACHE_DIR}")
    print(f"Save path    : {SAVE_PATH}")
    print(f"Batch size   : {GEN_BATCH_SIZE}")
    print("=" * 80)

    dataset = load_from_disk(DATASET_PATH)
    print(dataset)

    model, feature_extractor, tokenizer = load_model_and_processor()

    verified = verify_one_sample(
        dataset=dataset,
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        split_name=None,
        index=0,
    )

    if not verified and STOP_IF_VERIFY_EMPTY:
        raise RuntimeError(
            "One-sample verification failed. Full generation stopped."
        )

    if VERIFY_ONLY:
        print("VERIFY_ONLY=True, exiting after one-sample verification.")
        return

    for split in SPLITS_TO_PROCESS:
        if split not in dataset:
            continue

        dataset[split] = generate_split_with_cache(
            dataset_split=dataset[split],
            split_name=split,
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    print("\n" + "=" * 80)
    print(f"Saving final dataset to: {SAVE_PATH}")
    dataset.save_to_disk(SAVE_PATH)
    print("[SAVE] Done.")

    # Remove cache only after successful save_to_disk().
    remove_cache_dir()

    print("=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
