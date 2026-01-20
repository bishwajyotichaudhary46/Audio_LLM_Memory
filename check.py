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
from  MemoryModule.conponents.custom_whisper_mem import WhisperForConditionalGeneration as WhisperCustom

device = 'cpu' if torch.cuda.is_available() else 'cpu'
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", task="transcribe")

model_original = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model_mem = WhisperCustom.from_pretrained("openai/whisper-small").to(device)



# with torch.no_grad():
#     model_mem.model.decoder.layers[5].router_proj.weight.zero_()
#     model_mem.model.decoder.layers[5].router_proj.bias.fill_(-10.0)

#     model_mem.model.decoder.layers[5].router_proj.weight.requires_grad = False
#     model_mem.model.decoder.layers[5].router_proj.bias.requires_grad = False

#     model_mem.model.decoder.layers[7].router_proj.weight.zero_()
#     model_mem.model.decoder.layers[7].router_proj.bias.fill_(-10.0)

#     model_mem.model.decoder.layers[7].router_proj.weight.requires_grad = False
#     model_mem.model.decoder.layers[7].router_proj.bias.requires_grad = False
    

# for name, buf in model_mem.model.decoder.layers[2].named_buffers():
#     print(f"model.decoder.layers.2.{name} {buf.shape}")

#mem = model_mem.model.decoder.layers[2].context_mom.mult_mem[0].M

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_mem,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float32,
    device=device,
)
# pipe.model.generation_config.forced_decoder_ids = None

pipe_original = pipeline(
    "automatic-speech-recognition",
    model=model_original,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float32,
    device='cpu',
)
result_original = pipe_original('Data/train/Voice705.wav')
print(f"result_original: {result_original}")


result_change = pipe(
    "Data/train/Voice705.wav" 
)

print(f"result_change: {result_change}")
