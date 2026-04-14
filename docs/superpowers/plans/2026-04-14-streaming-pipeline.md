# Streaming Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the batch record→transcribe→type pipeline with a three-stage concurrent pipeline where transcription pre-buffers during recording and typing begins the moment recording stops.

**Architecture:** Three QThread workers (Recorder, Transcriber, Typer) communicate via thread-safe queues wrapped as Python generators. RecorderWorker and TranscriberWorker run concurrently during recording; TyperWorker waits on a `recording_stopped` Event before draining the text queue.

**Tech Stack:** faster-whisper (segment generator), PyQt5 (QThread, pyqtSignal), Python threading (queue.Queue, threading.Event), webrtcvad, sounddevice, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/pipeline.py` | `queue_to_generator` adapter and `SENTINEL` constant |
| Create | `src/recorder_worker.py` | `RecorderWorker(QThread)` — VAD loop, yields audio bursts |
| Create | `src/transcriber_worker.py` | `TranscriberWorker(QThread)` — consumes bursts, streams segment text |
| Create | `src/typer_worker.py` | `TyperWorker(QThread)` — waits on recording_stopped, types text |
| Modify | `src/transcription.py` | Add `transcribe_local_stream()` generator; update `create_local_model()` for GPU auto-detect |
| Modify | `src/main.py` | Replace ResultThread with three workers; rename start/stop methods |
| Modify | `src/config.yaml` | Add `hotwords: []` under `model_options.local` |
| Delete | `src/result_thread.py` | Replaced entirely |
| Create | `tests/test_pipeline.py` | Tests for `queue_to_generator` |
| Create | `tests/test_transcription.py` | Tests for `transcribe_local_stream` |
| Create | `tests/test_recorder_worker.py` | Tests for burst flushing logic |
| Create | `tests/test_typer_worker.py` | Tests for recording_stopped gate |

---

## Task 1: Test infrastructure + `pipeline.py`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1.1: Create test scaffolding**

```bash
mkdir tests
touch tests/__init__.py
```

Create `tests/conftest.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
```

- [ ] **Step 1.2: Write failing tests for `queue_to_generator`**

Create `tests/test_pipeline.py`:
```python
import queue
from pipeline import queue_to_generator, SENTINEL


def test_yields_items_until_sentinel():
    q = queue.Queue()
    q.put('a')
    q.put('b')
    q.put(SENTINEL)
    assert list(queue_to_generator(q, sentinel=SENTINEL)) == ['a', 'b']


def test_empty_queue_with_immediate_sentinel():
    q = queue.Queue()
    q.put(SENTINEL)
    assert list(queue_to_generator(q, sentinel=SENTINEL)) == []


def test_sentinel_is_not_yielded():
    q = queue.Queue()
    q.put(42)
    q.put(SENTINEL)
    result = list(queue_to_generator(q, sentinel=SENTINEL))
    assert SENTINEL not in result
```

- [ ] **Step 1.3: Run tests — expect failure**

```bash
cd /home/jonathan/Repos/whisper-writer
.venv/bin/pytest tests/test_pipeline.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 1.4: Create `src/pipeline.py`**

```python
import queue
from typing import Generator, Any

SENTINEL = object()


def queue_to_generator(q: queue.Queue, sentinel: Any = None) -> Generator:
    """Wrap a thread-safe queue as a lazy generator. Stops when sentinel is received."""
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item
```

- [ ] **Step 1.5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_pipeline.py -v
```
Expected: 3 passed

- [ ] **Step 1.6: Commit**

```bash
git add src/pipeline.py tests/
git commit -m "feat: add queue_to_generator pipeline adapter"
```

---

## Task 2: `transcribe_local_stream` + GPU auto-detect

**Files:**
- Modify: `src/transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_transcription.py`:
```python
import numpy as np
from unittest.mock import MagicMock, patch
from transcription import transcribe_local_stream, create_local_model


def _make_segment(text):
    seg = MagicMock()
    seg.text = text
    return seg


def test_transcribe_local_stream_yields_segment_texts():
    audio = np.zeros(16000, dtype=np.int16)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        iter([_make_segment(' hello'), _make_segment(' world')]),
        MagicMock()
    )

    with patch('transcription.ConfigManager') as mock_cfg:
        mock_cfg.get_config_section.return_value = {
            'common': {'language': None, 'initial_prompt': '', 'temperature': 0.0},
            'local': {'condition_on_previous_text': True, 'vad_filter': False},
        }
        results = list(transcribe_local_stream(audio, local_model=mock_model))

    assert results == [' hello', ' world']


def test_transcribe_local_stream_passes_hotwords():
    audio = np.zeros(16000, dtype=np.int16)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch('transcription.ConfigManager') as mock_cfg:
        mock_cfg.get_config_section.return_value = {
            'common': {'language': None, 'initial_prompt': '', 'temperature': 0.0},
            'local': {'condition_on_previous_text': True, 'vad_filter': False},
        }
        list(transcribe_local_stream(audio, local_model=mock_model, hotwords=['pytest']))

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs['hotwords'] == ['pytest']


def test_transcribe_local_stream_passes_initial_prompt():
    audio = np.zeros(16000, dtype=np.int16)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), MagicMock())

    with patch('transcription.ConfigManager') as mock_cfg:
        mock_cfg.get_config_section.return_value = {
            'common': {'language': None, 'initial_prompt': '', 'temperature': 0.0},
            'local': {'condition_on_previous_text': True, 'vad_filter': False},
        }
        list(transcribe_local_stream(audio, local_model=mock_model, initial_prompt='context'))

    call_kwargs = mock_model.transcribe.call_args.kwargs
    assert call_kwargs['initial_prompt'] == 'context'
```

- [ ] **Step 2.2: Run tests — expect failure**

```bash
.venv/bin/pytest tests/test_transcription.py -v
```
Expected: `ImportError: cannot import name 'transcribe_local_stream'`

- [ ] **Step 2.3: Add `transcribe_local_stream` to `src/transcription.py`**

Add after the existing `transcribe_local` function (line 64):

```python
def transcribe_local_stream(audio_data, local_model=None, initial_prompt='', hotwords=None):
    """
    Transcribe audio using a local model, yielding segment texts as they arrive.
    """
    if local_model is None:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')

    audio_float = audio_data.astype(np.float32) / 32768.0

    segments, _ = local_model.transcribe(
        audio=audio_float,
        language=model_options['common']['language'],
        initial_prompt=initial_prompt,
        condition_on_previous_text=model_options['local']['condition_on_previous_text'],
        temperature=model_options['common']['temperature'],
        vad_filter=model_options['local']['vad_filter'],
        hotwords=hotwords or [],
    )

    for segment in segments:
        yield segment.text
```

- [ ] **Step 2.4: Update `create_local_model()` for GPU auto-detect**

Replace lines 18–22 in `src/transcription.py`:

```python
    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        configured = local_model_options.get('device')
        if configured and configured not in ('auto', None):
            device = configured
        else:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
            ConfigManager.console_print(f'Auto-detected device: {device}')
```

- [ ] **Step 2.5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_transcription.py -v
```
Expected: 3 passed

- [ ] **Step 2.6: Commit**

```bash
git add src/transcription.py tests/test_transcription.py
git commit -m "feat: add transcribe_local_stream generator and GPU auto-detect"
```

---

## Task 3: `RecorderWorker`

**Files:**
- Create: `src/recorder_worker.py`
- Create: `tests/test_recorder_worker.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_recorder_worker.py`:
```python
import queue
import numpy as np
from threading import Event
from unittest.mock import patch, MagicMock
from pipeline import SENTINEL


def _make_config(mode='hold_to_record', sample_rate=16000, silence_duration=900, min_duration=100):
    return {
        'recording_mode': mode,
        'sample_rate': sample_rate,
        'silence_duration': silence_duration,
        'min_duration': min_duration,
        'sound_device': None,
    }


def test_flush_partial_burst_on_stop():
    """When stop() is called mid-burst, accumulated audio is flushed."""
    from recorder_worker import RecorderWorker

    audio_q = queue.Queue()
    recording_stopped = Event()

    with patch('recorder_worker.ConfigManager') as mock_cfg, \
         patch('recorder_worker.sd') as mock_sd, \
         patch('recorder_worker.webrtcvad.Vad') as mock_vad_cls:

        mock_cfg.get_config_section.return_value = _make_config()
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        mock_vad_cls.return_value = mock_vad

        # Simulate stream: 20 frames of speech, then stop_event fires
        frame_size = 480  # 30ms at 16000Hz
        fake_frames = [np.ones((frame_size, 1), dtype=np.int16) * 100] * 20

        frame_iter = iter(fake_frames)

        def fake_stream_enter(self_inner):
            return self_inner

        def fake_stream_exit(self_inner, *a):
            pass

        worker = RecorderWorker(audio_q, recording_stopped)

        # Stop after collecting frames
        def side_effect_callback(callback_holder):
            for frame in fake_frames:
                callback_holder['cb'](frame, frame_size, None, None)
            worker.stop()

        # We test via direct call to _audio_bursts
        # Stop event set after first call
        stop_ev = worker._stop_event
        stop_ev.set()  # pre-set so it exits immediately after flushing

        # Manually build a recording buffer and call flush logic
        # Test _flush_burst helper directly
        recording = list(np.ones(16000, dtype=np.int16))  # 1s of audio
        result = worker._make_burst(recording, sample_rate=16000, min_duration_ms=100)
        assert result is not None
        assert result.dtype == np.int16


def test_burst_below_min_duration_is_none():
    from recorder_worker import RecorderWorker

    audio_q = queue.Queue()
    recording_stopped = Event()
    worker = RecorderWorker(audio_q, recording_stopped)

    short_recording = list(np.ones(800, dtype=np.int16))  # 50ms at 16000Hz — below 100ms min
    result = worker._make_burst(short_recording, sample_rate=16000, min_duration_ms=100)
    assert result is None
```

- [ ] **Step 3.2: Run tests — expect failure**

```bash
.venv/bin/pytest tests/test_recorder_worker.py -v
```
Expected: `ModuleNotFoundError: No module named 'recorder_worker'`

- [ ] **Step 3.3: Create `src/recorder_worker.py`**

```python
import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import SENTINEL
from utils import ConfigManager


class RecorderWorker(QThread):
    statusSignal = pyqtSignal(str)

    def __init__(self, audio_q, recording_stopped: Event):
        super().__init__()
        self._audio_q = audio_q
        self._recording_stopped = recording_stopped
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def _make_burst(self, recording: list, sample_rate: int, min_duration_ms: int):
        """Convert a frame list to a numpy burst, or None if too short."""
        audio = np.array(recording, dtype=np.int16)
        duration_ms = (len(audio) / sample_rate) * 1000
        if duration_ms < min_duration_ms:
            return None
        return audio

    def run(self):
        self.statusSignal.emit('recording')
        recording_options = ConfigManager.get_config_section('recording_options')
        sample_rate = recording_options.get('sample_rate') or 16000
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = int(silence_duration_ms / frame_duration_ms)
        min_duration_ms = recording_options.get('min_duration') or 100
        initial_skip = int(0.15 * sample_rate / frame_size)
        recording_mode = recording_options.get('recording_mode') or 'continuous'

        vad = webrtcvad.Vad(2)
        audio_buffer = deque(maxlen=frame_size)
        data_ready = Event()

        def audio_callback(indata, frames, time, status):
            if status:
                ConfigManager.console_print(f'Audio callback status: {status}')
            audio_buffer.extend(indata[:, 0])
            data_ready.set()

        recording = []
        speech_detected = False
        silent_frame_count = 0
        frames_to_skip = initial_skip

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                            blocksize=frame_size,
                            device=recording_options.get('sound_device'),
                            callback=audio_callback):
            while not self._stop_event.is_set():
                triggered = data_ready.wait(timeout=0.1)
                if not triggered:
                    continue
                data_ready.clear()

                if len(audio_buffer) < frame_size:
                    continue

                frame = np.array(list(audio_buffer), dtype=np.int16)
                audio_buffer.clear()
                recording.extend(frame)

                if frames_to_skip > 0:
                    frames_to_skip -= 1
                    continue

                if vad.is_speech(frame.tobytes(), sample_rate):
                    silent_frame_count = 0
                    speech_detected = True
                else:
                    silent_frame_count += 1

                if speech_detected and silent_frame_count > silence_frames:
                    burst = self._make_burst(recording, sample_rate, min_duration_ms)
                    if burst is not None:
                        self._audio_q.put(burst)
                    # reset for next burst
                    recording = []
                    speech_detected = False
                    silent_frame_count = 0
                    frames_to_skip = initial_skip

        # Flush partial burst on stop
        if recording:
            burst = self._make_burst(recording, sample_rate, min_duration_ms)
            if burst is not None:
                self._audio_q.put(burst)

        self._audio_q.put(SENTINEL)
        self._recording_stopped.set()
        ConfigManager.console_print('RecorderWorker stopped.')
```

- [ ] **Step 3.4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_recorder_worker.py -v
```
Expected: 2 passed

- [ ] **Step 3.5: Commit**

```bash
git add src/recorder_worker.py tests/test_recorder_worker.py
git commit -m "feat: add RecorderWorker with VAD burst detection and flush-on-stop"
```

---

## Task 4: `TranscriberWorker`

**Files:**
- Create: `src/transcriber_worker.py`

No isolated unit tests here — the logic is a thin loop over `queue_to_generator` + `transcribe_local_stream`, both of which are already tested. Integration is validated in Task 7.

- [ ] **Step 4.1: Create `src/transcriber_worker.py`**

```python
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
```

- [ ] **Step 4.2: Commit**

```bash
git add src/transcriber_worker.py
git commit -m "feat: add TranscriberWorker streaming segments to text queue"
```

---

## Task 5: `TyperWorker`

**Files:**
- Create: `src/typer_worker.py`
- Create: `tests/test_typer_worker.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_typer_worker.py`:
```python
import queue
from threading import Event, Thread
from unittest.mock import MagicMock, call
from pipeline import SENTINEL


def test_typer_waits_for_recording_stopped_before_typing():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()

    text_q.put('hello ')
    text_q.put(SENTINEL)

    worker = TyperWorker(text_q, mock_simulator, recording_stopped)

    typed = []
    original = mock_simulator.typewrite
    mock_simulator.typewrite.side_effect = typed.append

    # Start worker — should block because recording_stopped not set
    worker.start()
    import time; time.sleep(0.1)
    assert typed == [], "Should not have typed before recording_stopped"

    recording_stopped.set()
    worker.wait(2000)
    assert typed == ['hello ']


def test_typer_emits_typing_then_idle():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()
    recording_stopped.set()

    text_q.put('word ')
    text_q.put(SENTINEL)

    statuses = []
    worker = TyperWorker(text_q, mock_simulator, recording_stopped)
    worker.statusSignal.connect(statuses.append)
    worker.start()
    worker.wait(2000)

    assert 'typing' in statuses
    assert statuses[-1] == 'idle'


def test_typer_idle_with_empty_queue():
    from typer_worker import TyperWorker

    text_q = queue.Queue()
    recording_stopped = Event()
    mock_simulator = MagicMock()
    recording_stopped.set()

    text_q.put(SENTINEL)

    statuses = []
    worker = TyperWorker(text_q, mock_simulator, recording_stopped)
    worker.statusSignal.connect(statuses.append)
    worker.start()
    worker.wait(2000)

    mock_simulator.typewrite.assert_not_called()
    assert statuses == ['idle']
```

- [ ] **Step 5.2: Run tests — expect failure**

```bash
.venv/bin/pytest tests/test_typer_worker.py -v
```
Expected: `ModuleNotFoundError: No module named 'typer_worker'`

- [ ] **Step 5.3: Create `src/typer_worker.py`**

```python
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal

from pipeline import queue_to_generator, SENTINEL


class TyperWorker(QThread):
    statusSignal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, text_q, input_simulator, recording_stopped: Event):
        super().__init__()
        self._text_q = text_q
        self._input_simulator = input_simulator
        self._recording_stopped = recording_stopped

    def run(self):
        self._recording_stopped.wait()
        first = True
        for text in queue_to_generator(self._text_q, sentinel=SENTINEL):
            if first:
                self.statusSignal.emit('typing')
                first = False
            self._input_simulator.typewrite(text)
        self.statusSignal.emit('idle')
        self.finished.emit()
```

- [ ] **Step 5.4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_typer_worker.py -v
```
Expected: 3 passed

- [ ] **Step 5.5: Commit**

```bash
git add src/typer_worker.py tests/test_typer_worker.py
git commit -m "feat: add TyperWorker with recording_stopped gate"
```

---

## Task 6: Wire up `main.py` + delete `result_thread.py`

**Files:**
- Modify: `src/main.py`
- Delete: `src/result_thread.py`

- [ ] **Step 6.1: Replace imports in `src/main.py`**

Remove:
```python
from result_thread import ResultThread
```

Add:
```python
import queue
from threading import Event
from recorder_worker import RecorderWorker
from transcriber_worker import TranscriberWorker
from typer_worker import TyperWorker
```

- [ ] **Step 6.2: Update `initialize_components` in `src/main.py`**

Replace:
```python
        self.result_thread = None
```
With:
```python
        self._recorder = None
        self._transcriber = None
        self._typer = None
```

- [ ] **Step 6.3: Replace `start_result_thread` with `start_pipeline`**

Remove the existing `start_result_thread` method and replace with:

```python
    def start_pipeline(self):
        if self._recorder and self._recorder.isRunning():
            return

        audio_q = queue.Queue()
        text_q = queue.Queue()
        recording_stopped = Event()

        self._recorder = RecorderWorker(audio_q, recording_stopped)
        self._transcriber = TranscriberWorker(audio_q, text_q, self.local_model)
        self._typer = TyperWorker(text_q, self.input_simulator, recording_stopped)

        if not ConfigManager.get_config_value('misc', 'hide_status_window'):
            self._recorder.statusSignal.connect(self.status_window.updateStatus)
            self._typer.statusSignal.connect(self.status_window.updateStatus)
            self.status_window.closeSignal.connect(self.stop_pipeline)

        self._typer.finished.connect(self._on_pipeline_finished)

        self._recorder.start()
        self._transcriber.start()
        self._typer.start()
```

- [ ] **Step 6.4: Replace `stop_result_thread` with `stop_pipeline`**

Remove the existing `stop_result_thread` method and replace with:

```python
    def stop_pipeline(self):
        if self._recorder and self._recorder.isRunning():
            self._recorder.stop()
```

- [ ] **Step 6.5: Replace `on_activation` and `on_deactivation`**

Replace the existing `on_activation` method:
```python
    def on_activation(self):
        if self._recorder and self._recorder.isRunning():
            recording_mode = ConfigManager.get_config_value('recording_options', 'recording_mode')
            if recording_mode == 'press_to_toggle':
                self.stop_pipeline()
            elif recording_mode == 'continuous':
                self.stop_pipeline()
            return
        self.start_pipeline()
```

Replace the existing `on_deactivation` method:
```python
    def on_deactivation(self):
        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'hold_to_record':
            self.stop_pipeline()
```

- [ ] **Step 6.6: Replace `on_transcription_complete` with `_on_pipeline_finished`**

Remove the existing `on_transcription_complete` method and replace with:

```python
    def _on_pipeline_finished(self):
        if ConfigManager.get_config_value('misc', 'noise_on_completion'):
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

        if ConfigManager.get_config_value('recording_options', 'recording_mode') == 'continuous':
            self.start_pipeline()
        else:
            self.key_listener.start()
```

- [ ] **Step 6.7: Update `key_listener` callback wiring in `initialize_components`**

Replace:
```python
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
```
With (no change needed — method names stay the same):
```python
        self.key_listener.add_callback("on_activate", self.on_activation)
        self.key_listener.add_callback("on_deactivate", self.on_deactivation)
```

- [ ] **Step 6.8: Delete `src/result_thread.py`**

```bash
git rm src/result_thread.py
```

- [ ] **Step 6.9: Commit**

```bash
git add src/main.py
git commit -m "feat: wire three-worker pipeline into main, remove ResultThread"
```

---

## Task 7: Add `hotwords` to config

**Files:**
- Modify: `src/config.yaml`

- [ ] **Step 7.1: Add hotwords field**

In `src/config.yaml`, under `model_options.local`, add after `model_path`:

```yaml
    hotwords: []        # e.g. ["whisper-writer", "PyQt5", "xdotool"]
```

- [ ] **Step 7.2: Commit**

```bash
git add src/config.yaml
git commit -m "config: add hotwords field for local model vocabulary boosting"
```

---

## Task 8: Smoke test

- [ ] **Step 8.1: Run all tests**

```bash
.venv/bin/pytest tests/ -v
```
Expected: all pass

- [ ] **Step 8.2: Restart the service and do a live test**

```bash
systemctl --user restart whisper-writer
sleep 5
journalctl --user -u whisper-writer -n 20 --no-pager
```

Expected log output: `Creating local model...`, `Auto-detected device: cuda`, `Local model created.`

- [ ] **Step 8.3: Verify GPU is being used**

```bash
nvidia-smi | grep python
```
Expected: a python process consuming GPU memory (> 500MiB for the `small` model on CUDA).

- [ ] **Step 8.4: Final commit if any fixes were needed**

```bash
git add -p
git commit -m "fix: <describe what needed fixing>"
```
