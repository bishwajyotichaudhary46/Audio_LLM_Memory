
def preprocess(batch, feature_extractor, tokenizer):
    waveform= batch['audio_path']['array']

    batch['input_features'] = feature_extractor(waveform, sampling_rate=16000).input_features[0]

    batch['labels'] = tokenizer(batch['sentence']).input_ids

    return batch