import os
import asyncio
import logging

from src.asr.asr_factory import ASRFactory

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    asr_type = os.environ.get('ASR_TYPE') or 'faster_whisper'
    asr_model_type = os.environ.get('ASR_MODEL_TYPE') or 'large-v3'

    asr_pipeline = ASRFactory.create_asr_pipeline(asr_type, **{'model_type': asr_model_type})

    transcription = asr_pipeline.transcribe_file("audio_debug/1e8df508b000d1a5df741db9284c858e_127446.wav", "zh")
    print(transcription)
