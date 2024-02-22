import tornado.ioloop
import tornado.web
import logging
import asyncio
import os
import time
import random
import string
import base64
import openai
from openai import AsyncOpenAI
import torch
from src.asr.asr_factory import ASRFactory
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf


class VoiceChatRequest(tornado.web.RequestHandler):

    def __init__(self, *args, **kwargs):
        super(VoiceChatRequest, self).__init__(*args, **kwargs)
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Access-Control-Allow-Origin', "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        # 请求方式
        self.set_header("Access-Control-Allow-Methods", "*")

        self.base_url = os.environ.get("OPENAI_BASE_URL")

    async def write_message(self, key):
        for idx in range(10):
            val = {"code": 200, "msg": "success", "data": f"{key} {idx}"}
            self.write(f'data:{val}\n\n')
            await self.flush()
            await asyncio.sleep(3)
        await asyncio.sleep(1)
        return ''

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

    async def post(self):
        logging.info('new post request connect')
        # 接收文件
        file = self.request.files['file'][0]
        logging.debug(f'file received {file}')
        original_fname = file['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
        final_filename = fname+extension
        file_path = final_filename
        output_file = open(file_path, 'wb')
        output_file.write(file['body'])
        if not os.path.exists(file_path):
            logging.warning('audio file save failed')
            val = {"code": 400, "msg": 'audio file save failed'}
            self.write(f'data:{val}\n\n')
            await self.flush()
            return

        # 解析音频到文字
        start = time.time()
        transcription = await asr_pipeline.transcribe_file(file_path, None)
        if 'text' in transcription and transcription['text'] != '':
            end = time.time()
            text = transcription["text"]
            logging.debug(f'解析到声音内容： {text}， 耗时: {end - start}')

            messages = [{"role": "user", "content": text}]
            content = await self.get_chat_response_sync(messages)
            val = {"code": 200, "msg": 'success', 'text': content}
            self.write(f'data:{val}\n\n')
            await self.flush()

            client = openai.OpenAI()
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=content,
            )

            for data in response.iter_bytes():
                # logging.debug(f'speech component: {data}')
                bytes_str = base64.b64encode(data).decode('utf-8')
                val = {"code": 200, "msg": 'success', 'audio': bytes_str}
                self.write(f'data:{val}\n\n')
                await self.flush()

            # speech = synthesiser("Hello, my dog is cooler than you!",
            #                      forward_params={"speaker_embeddings": speaker_embedding})
            # target_file = f"speech{random.randint(1000, 9999)}.wav"
            # sf.write(target_file, speech["audio"], samplerate=speech["sampling_rate"])
            # if not os.path.exists(target_file):
            #     logging.warning('convert text to speech failed')
            #     val = {"code": 500, "msg": 'convert text to speech failed'}
            #     self.write(f'data:{val}\n\n')
            #     return
            #
            # with open(target_file, mode='rb') as f:
            #     byte_array = f.read()
            #     bytes_str = base64.b64encode(byte_array).decode('utf-8')
            #     val = {"code": 200, "msg": 'success', 'text': bytes_str}
            #     self.write(f'data:{val}\n\n')
            #     await self.flush()
            # # 删掉临时文件
            # os.remove(target_file)


def make_app():
    return tornado.web.Application([
        (r"/cogen/v1/voice_chat", VoiceChatRequest)
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    app = make_app()

    asr_args = {"model_size": "large-v3"}
    asr_pipeline = ASRFactory.create_asr_pipeline('faster_whisper', **asr_args)

    # synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    #
    # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    app.listen(6006)
    logging.info("sse服务启动")
    tornado.ioloop.IOLoop.current().start()