import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
)
import evaluate


# Configuration


MODEL_ID = "openai/whisper-small"
DATASET_NAME = "google/fleurs"
DATASET_CONFIG = "bn_in"
SPLIT = "test"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load Model & Processor


processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained("artifacts/checkpoint/lora_dpo/merged_checkpoint_24000")

model.to(device)
model.eval()


# Load a valid Whisper GenerationConfig
generation_config = GenerationConfig.from_pretrained(
    "openai/whisper-small"
)

generation_config.language = "bn"
generation_config.task = "transcribe"

generation_config.forced_decoder_ids = (
    processor.get_decoder_prompt_ids(
        language="bn",
        task="transcribe",
    )
)

# Some community checkpoints contain empty suppress_tokens.
generation_config.suppress_tokens = None
generation_config.begin_suppress_tokens = None

model.generation_config = generation_config


# Load Dataset
dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG,
    split=SPLIT,
)

print(dataset)


# Prediction Function
def map_to_pred(batch):

    audio = batch["audio"]

    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    ).input_features.to(device)

    with torch.no_grad():

        predicted_ids = model.generate(
            input_features,
            max_new_tokens=225,
        )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    batch["prediction"] = transcription

    return batch



# Run Inference
print("Running inference...")

result = dataset.map(
    map_to_pred,
    desc="Transcribing",
)


# Evaluation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

wer = wer_metric.compute(
    predictions=result["prediction"],
    references=result["transcription"],
)

cer = cer_metric.compute(
    predictions=result["prediction"],
    references=result["transcription"],
)

print("=" * 60)
print(f"WER : {wer:.4f}")
print(f"CER : {cer:.4f}")
print("=" * 60)