import os
import asyncio
import json
import time
import openai
from openai import AsyncOpenAI
import logging

from .buffering_strategy_interface import BufferingStrategyInterface

class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with silence detection.

    This class is responsible for handling audio chunks, detecting silence at the end of each chunk,
    and initiating the transcription process for the chunk.

    Attributes:
        client (Client): The client instance associated with this buffering strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering strategy.
            **kwargs: Additional keyword arguments, including 'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client

        self.chunk_length_seconds = os.environ.get('BUFFERING_CHUNK_LENGTH_SECONDS')
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = kwargs.get('chunk_length_seconds')
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get('BUFFERING_CHUNK_OFFSET_SECONDS')
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = kwargs.get('chunk_offset_seconds')
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.error_if_not_realtime = os.environ.get('ERROR_IF_NOT_REALTIME')
        if not self.error_if_not_realtime:
            self.error_if_not_realtime = kwargs.get('error_if_not_realtime', False)
        
        self.processing_flag = False

        self.base_url = os.environ.get("OPENAI_BASE_URL")

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process audio chunks by checking their length and scheduling asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk length and, if so,
        it schedules asynchronous processing of the audio.

        Args:
            websocket (Websocket): The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        chunk_length_in_bytes = self.chunk_length_seconds * self.client.sampling_rate * self.client.samples_width
        if len(self.client.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                exit("Error in realtime processing: tried processing a new chunk while the previous one was still being processed")

            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task
            asyncio.create_task(self.process_audio_async(websocket, vad_pipeline, asr_pipeline))

    async def get_chat_response(self, messages) -> str:
        result = []
        try:
            aclient = AsyncOpenAI(base_url=self.base_url)
            response = await aclient.chat.completions.create(model='gpt-3.5-turbo', messages=messages,
                                                             temperature=1, max_tokens=2048, stream=True)
            async for chunk in response:
                chunk_message = chunk.choices[0].delta  # extract the message
                if chunk_message.content is not None:
                    chunk_content = chunk_message.content
                    result.append(chunk_content)
        except openai.APIError as e:
            logging.warning(f"openai api error {repr(e)}")
        except Exception as e:
            logging.warning(f'query from openai error {repr(e)}')
        result = ''.join(result)
        logging.info(f'request {messages} \nchat response {result}')
        return result
    
    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity detection and transcription of
        the audio data. It sends the transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """   
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)

        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            return

        last_segment_should_end_before = ((len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width)) - self.chunk_offset_seconds)
        if vad_results[-1]['end'] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.client)
            if transcription['text'] != '':
                end = time.time()
                transcription['processing_time'] = end - start
                json_transcription = json.dumps(transcription) 
                await websocket.send(json_transcription)

                # 从GPT获取数据
                start = time.time()
                messages = [{"role": "user", "content": transcription['text']}]
                content = await self.get_chat_response(messages)
                end = time.time()
                result = {'text': content, 'processing_time': end - start}
                json_transcription = json.dumps(result)
                logging.info(f'set content to client {json_transcription}')
                await websocket.send(json_transcription)

            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()
        
        self.processing_flag = False
