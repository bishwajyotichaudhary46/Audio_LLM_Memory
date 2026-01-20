from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
import evaluate
# Custom Whisper memory model
from MemoryModule.conponents.custom_whisper_mem import WhisperForConditionalGeneration as WhisperCustom

# Device configuration
device = torch.device("cpu")

# Load feature extractor and tokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")

# Load model on CPU

model_mem = WhisperCustom.from_pretrained("openai/whisper-small").to(device)

# Load and preprocess dataset

dataset = load_dataset(
    "csv",
    data_dir="Data",
    data_files="train.csv"
)

# Prepare audio file paths
dataset = dataset.map(lambda x: {"audio_path": f"Data/train/{x['audio_id']}.wav"})

# Cast audio column to numerical array
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

# Split train/test
dataset = dataset["train"].train_test_split(test_size=0.1)

# Preprocessing function
def preprocess(batch):
    waveform = batch['audio_path']['array']
    batch['input_features'] = feature_extractor(waveform, sampling_rate=16000).input_features[0]
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    return batch

# Map preprocessing without moving to device
dataset = dataset.map(
    preprocess,
    remove_columns=dataset["train"].column_names
)


# Data collator for padding

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Prepare input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # Prepare labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if needed
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    decoder_start_token_id=model_mem.config.decoder_start_token_id,
)


# Freeze some layers

for p in model_mem.parameters():
    p.requires_grad = False

# unfreeze decoder layers 5 and 7
for layer_idx in [5, 7]:
    for p in model_mem.model.decoder.layers[layer_idx].context_mom.parameters():
        p.requires_grad = True
    for p in model_mem.model.decoder.layers[layer_idx].router_proj.parameters():
        p.requires_grad = True
    
    for p in model_mem.model.decoder.layers[layer_idx].memory_norm.parameters():
        p.requires_grad = True



# Set generation config

model_mem.generation_config.language = "ne"
model_mem.generation_config.task = "transcribe"
model_mem.config.forced_decoder_ids = None
model_mem.config.suppress_tokens = None


# Evaluation metric

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Training arguments (CPU)

training_args = Seq2SeqTrainingArguments(
    output_dir="trained_check_point",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=3000,
    gradient_checkpointing=False,
    fp16=False,  # CPU cannot use fp16
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=250,
    eval_steps=25,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=True,
    push_to_hub=False,
    no_cuda=True
)

#print('start trained')
# Trainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model_mem,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)


# Start training

trainer.train()



# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model_mem,
#     tokenizer=tokenizer,
#     feature_extractor=feature_extractor,
#     torch_dtype=torch.float32,
#     device='cpu',
# )
# pipe_original = pipeline(
#     "automatic-speech-recognition",
#     model=model_original,
#     tokenizer=tokenizer,
#     feature_extractor=feature_extractor,
#     torch_dtype=torch.float32,
#     device='cpu',
# )
# result_original = pipe_original('Data/test/Voice118.wav')
# print(f"result_original: {result_original}")
# result_change = pipe('Data/test/Voice118.wav')
# print(f"result_change: {result_original}")



