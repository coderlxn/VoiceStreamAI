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
        logging.debug("读取 OPENAI_BASE_URL = " + self.base_url)

    async def process_audio(self, websocket, vad_pipeline, asr_pipeline):
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
                # exit("Error in realtime processing: tried processing a new chunk while the previous one was still
                # being processed")
                logging.debug("上个转换任务还没有结束，跳过本次！！！")
                return

            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task
            # asyncio.create_task(self.process_audio_async(websocket, vad_pipeline, asr_pipeline))
            await self.process_audio_async(websocket, vad_pipeline, asr_pipeline)

            # loop = asyncio.get_event_loop()
            # result = loop.run_until_complete(self.process_audio_async(websocket, vad_pipeline, asr_pipeline))

    async def get_chat_response(self, messages) -> str:
        result = []
        start = time.time()
        try:
            aclient = AsyncOpenAI(base_url=self.base_url)
            response = await aclient.chat.completions.create(model='gpt-3.5-turbo', messages=messages,
                                                             temperature=1, max_tokens=2048, stream=True)

            logging.debug("GPT 耗时0 = {:.3f} : s%".format((time.time() - start)), messages)
            async for chunk in response:
                chunk_message = chunk.choices[0].delta  # extract the message
                logging.debug("GPT 耗时1 = {:.3f}".format(time.time() - start))
                if chunk_message.content is not None:
                    chunk_content = chunk_message.content
                    result.append(chunk_content)
        except openai.APIError as e:
            logging.warning(f"openai api error {repr(e)}")
        except Exception as e:
            logging.warning(f'query from openai error {repr(e)}')
        logging.debug("GPT 耗时2 = {:.3f}".format(time.time() - start))
        result = ''.join(result)
        logging.info(f'request {messages} \nchat response {result}')
        logging.debug("GPT 耗时3 = {:.3f}".format(time.time() - start))

        return result

    async def get_chat_response_sync(self, messages) -> str:
        start = time.time()
        logging.debug("开始GPT请求")
        try:
            client = AsyncOpenAI(base_url=self.base_url)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            logging.debug("syncGPT 耗时2 = {:.3f}".format(time.time() - start))
            result = (await completion).choices[0].message.content
            logging.info(f'request {messages} \nchat response {result}')
            logging.debug("GPT请求完成 耗时3 = {:.3f}".format(time.time() - start))
            return result
        except Exception as e:
            return repr(e)
    
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
        logging.debug('检测到活动声音')

        last_segment_should_end_before = ((len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width)) - self.chunk_offset_seconds)
        if vad_results[-1]['end'] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.client)
            if transcription['text'] != '':
                end = time.time()
                logging.debug(f'解析到声音内容： {transcription["text"]}， 耗时: {end - start}')
                transcription['processing_time'] = end - start
                json_transcription = json.dumps(transcription) 
                await websocket.send(json_transcription)

                # 从GPT获取数据
                start = time.time()
                messages = [{"role": "user", "content": transcription['text']}]
                content = await self.get_chat_response(messages)
                # content = await self.get_chat_response_sync(messages)
                end = time.time()

                logging.debug("GPT 总耗时 = {:.3f}".format(end - start))

                result = {'ai_resp': content, 'text': content, 'processing_time': end - start}
                json_transcription = json.dumps(result)
                logging.info(f'set content to client {json_transcription}')
                await websocket.send(json_transcription)

            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()
        
        self.processing_flag = False
