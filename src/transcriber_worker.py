from PyQt5.QtCore import QThread

from pipeline import queue_to_generator, SENTINEL
from transcription import transcribe_local_stream, post_process_transcription
from utils import ConfigManager


class TranscriberWorker(QThread):

    def __init__(self, audio_q, text_q, local_model):
        super().__init__()
        self._audio_q = audio_q
        self._text_q = text_q
        self._local_model = local_model

    def run(self):
        model_options = ConfigManager.get_config_section('model_options')
        hotwords = model_options['local'].get('hotwords') or []
        initial_prompt = model_options['common'].get('initial_prompt') or ''
        last_raw = initial_prompt

        for burst in queue_to_generator(self._audio_q, sentinel=SENTINEL):
            try:
                for raw_text in transcribe_local_stream(
                    burst,
                    local_model=self._local_model,
                    initial_prompt=last_raw,
                    hotwords=hotwords,
                ):
                    processed = post_process_transcription(raw_text)
                    if processed:
                        self._text_q.put(processed)
                        last_raw = raw_text
            except Exception as e:
                ConfigManager.console_print(f'Transcription error: {e}')

        self._text_q.put(SENTINEL)
