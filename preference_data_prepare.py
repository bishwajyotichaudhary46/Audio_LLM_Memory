import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor
from MemoryModule.n_gram_attention_memory.one_gram_memory import (
    WhisperForConditionalGeneration as WhisperCustom
)

dataset = load_from_disk("artifacts/Data/indic_voice")

# Start small. Increase to 2 or 4 only if stable.
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

max_audio_seconds = 30
sampling_rate = 16000

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

model_path = "artifacts/checkpoint/n_gram_attention_memory/last_layer/sft/gram_one_memory_only/best_model"

model = WhisperCustom.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(device)

model.eval()

try:
    model.gradient_checkpointing_disable()
except Exception:
    pass

# For low-memory generation, disable cache.
# Slower, but safer for OOM.
model.config.use_cache = False
model.generation_config.use_cache = False


def clean_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_audio_array(item):
    frames = item.get_all_samples()
    arr = frames.data

    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[0] <= 8:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)

    arr = arr.squeeze()

    if arr is None or arr.size == 0:
        return None

    # Whisper supports 30 sec input. Truncate long audio.
    max_len = sampling_rate * max_audio_seconds
    if arr.shape[0] > max_len:
        arr = arr[:max_len]

    return arr


def generate_rejected(batch):
    audio_arrays = []
    valid_indices = []

    rejected = [""] * len(batch["audio"])

    for i, item in enumerate(batch["audio"]):
        try:
            arr = extract_audio_array(item)
            if arr is None:
                continue

            audio_arrays.append(arr)
            valid_indices.append(i)

        except Exception as e:
            print(f"[SKIP BAD AUDIO] batch index={i}, error={e}")
            continue

    if len(audio_arrays) == 0:
        return {"rejected": rejected}

    try:
        inputs = processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sampling_rate * max_audio_seconds,
        )

        input_features = inputs["input_features"].to(
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        with torch.inference_mode():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=128,   # 225 is high; increase later if needed
                num_beams=1,
                do_sample=False,
                task="transcribe",
                language="ne",
                return_timestamps=False,
                use_cache=False,
            )

        transcriptions = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )

        for idx, text in zip(valid_indices, transcriptions):
            rejected[idx] = text.strip()

    except torch.OutOfMemoryError as e:
        print("[OOM] Skipping this batch:", e)
        clean_cuda()

    finally:
        try:
            del inputs, input_features, predicted_ids
        except Exception:
            pass
        clean_cuda()

    return {"rejected": rejected}


dataset["train"] = dataset["train"].map(
    generate_rejected,
    batched=True,
    batch_size=batch_size,
    desc="Generating rejected transcriptions for train",
)

dataset["valid"] = dataset["valid"].map(
    generate_rejected,
    batched=True,
    batch_size=batch_size,
    desc="Generating rejected transcriptions for valid",
)

dataset.save_to_disk("artifacts/Data/indic_voice_dpo_onegram_memory")

print("saved")