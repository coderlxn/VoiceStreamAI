class ASRInterface:
    async def transcribe(self, client):
        """
        Transcribe the given audio data.

        :param client: The client object with all the member variables including the buffer
        :return: The transcription structure, see for example the faster_whisper_asr.py file.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def transcribe_file(self, file_path, language):
        """
        Transcribe the given audio data.

        :param file_path: The voice file
        :param language: The voice file
        :return: The transcription structure, see for example the faster_whisper_asr.py file.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
