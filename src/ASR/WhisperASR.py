from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Dataset, Audio
from typing import Union, Iterable
import torch


class WhisperASR:
    def __init__(self,
                 device=None,
                 max_new_tokens=128,
                 language='english',
                 task='transcribe',
                 chunk_length_s=30,
                 batch_size=16,
                 word_level_timestamps=False):
        self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = 'openai/whisper-large-v3'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
                                                          torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True,
                                                          use_safetensors=True)
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline('automatic-speech-recognition',
                             model=model,
                             tokenizer=processor.tokenizer,
                             feature_extractor=processor.feature_extractor,
                             chunk_length_s=chunk_length_s,
                             max_new_tokens=max_new_tokens,
                             batch_size=batch_size,
                             return_timestamps=True if not word_level_timestamps else 'word',
                             torch_dtype=torch_dtype,
                             device=self.device)
        self._kwargs = {'language': language, 'task': task}

    def __call__(self, audio_file: Union[str, Iterable]) -> Union[str, Iterable]:
        ds = Dataset.from_dict({'audio': [audio_file] if isinstance(audio_file, str) else audio_file}).cast_column(
            'audio', Audio(sampling_rate=16000))
        if isinstance(audio_file, str):
            return self._handle(ds)[0]
        else:
            return self._handle(ds)

    def _handle(self, ds: Dataset) -> list:
        res = []
        for data in ds:
            sample = data['audio']
            pred = self.pipe(sample.copy(), generate_kwargs=self._kwargs)
            res.append(pred['chunks'])
        return res
