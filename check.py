from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer,pipeline
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union,Tuple, Optional
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Trainer
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import librosa
import transformers
import os
import inspect
import evaluate
from  MemoryModule.conponents.modeling_whisper import ConditionalGeneration

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")

model_original = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = ConditionalGeneration.from_pretrained("openai/whisper-small")

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float32,
    device='cpu',
)
pipe_original = pipeline(
    "automatic-speech-recognition",
    model=model_original,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float32,
    device='cpu',
)
result_original = pipe_original('Data/male-female-data/male-female-data/Voice13.wav')
print(f"result_original: {result_original}")
result_change = pipe('Data/male-female-data/male-female-data/Voice13.wav')
print(f"result_change: {result_original}")



