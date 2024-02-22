import logging

from .whisper_asr import WhisperASR
from .faster_whisper_asr import FasterWhisperASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(type, **kwargs):
        if type == "whisper":
            logging.debug('use whisper asr')
            return WhisperASR(**kwargs)
        if type == "faster_whisper":
            logging.debug('use faster whisper asr')
            return FasterWhisperASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {type}")
