# Streaming Pipeline Design

**Date:** 2026-04-14  
**Status:** Approved

## Overview

Replace whisper-writer's batch pipeline (record all → transcribe all → type all) with a three-stage streaming pipeline. Recording and transcription run concurrently (transcription pre-buffers into a queue while recording is still active). Typing begins the moment recording stops, streaming smoothly from the pre-buffered queue.

---

## Architecture

### Pipeline Stages

```
audio_bursts(stop_event) ──► transcribe_segments(bursts) ──► type_stream(segments)
```

Each stage is a Python generator function. Concurrency between stages is provided by thread-safe queues wrapped as generators — threading is hidden inside the adapters; the pipeline composition remains pure generator chaining.

### Stage 1 — `audio_bursts(stop_event) -> Generator[np.ndarray]`

Located in `src/recorder_worker.py`.

- Runs the existing WebRTC VAD loop (30ms frames, configurable silence duration)
- When VAD detects silence after speech: yields the accumulated burst, resets, and begins the next burst immediately
- When `stop_event` is set (user stops recording): flushes whatever audio has been accumulated (if > `min_duration`) as a final burst, then returns — closing the generator and signalling downstream via queue sentinel
- Skips initial 150ms frames to avoid key-press noise (existing behaviour)
- Emits `statusSignal('recording')` when started

### Stage 2 — `transcribe_segments(bursts) -> Generator[str]`

Located in `src/transcriber_worker.py`.

- Iterates over the burst generator; for each burst calls `faster_whisper.transcribe()` with the segment generator (no `list()` materialisation)
- Yields each `segment.text` as it arrives from the model
- Carries the last segment's text as `initial_prompt` into the next burst for cross-burst context continuity
- Passes `hotwords` from config if set
- Applies existing post-processing (strip, trailing period, capitalisation) per segment

### Stage 3 — `type_stream(segments)`

Located in `src/typer_worker.py`.

- Waits on a `recording_stopped` event before consuming anything from `text_q`
- Once released, iterates over the segment generator, calling `input_simulator.typewrite(text)` for each segment
- Emits `statusSignal('typing')` on first segment, `statusSignal('idle')` when generator is exhausted

By the time `recording_stopped` fires, transcription has been running in the background and `text_q` is partially or fully pre-filled — so typing begins immediately with no perceivable delay.

---

## Concurrency Model

Each stage runs in its own `QThread`. Stages are connected by a thin adapter:

```python
def queue_to_generator(q: queue.Queue, sentinel=None):
    while True:
        item = q.get()
        if item is sentinel:
            return
        yield item
```

The RecorderWorker puts bursts onto `audio_q` and sets a shared `recording_stopped` event when it finishes. The TranscriberWorker reads from `audio_q` via `queue_to_generator` and puts segment text onto `text_q`. The TyperWorker waits on `recording_stopped`, then drains `text_q`. On stop, a `None` sentinel cascades through both queues to shut down all downstream workers cleanly.

---

## Component Changes

### Deleted
- `src/result_thread.py` — replaced entirely by the three worker modules below

### New Files
- `src/recorder_worker.py` — `RecorderWorker(QThread)` wrapping the `audio_bursts` generator
- `src/transcriber_worker.py` — `TranscriberWorker(QThread)` wrapping the `transcribe_segments` generator
- `src/typer_worker.py` — `TyperWorker(QThread)` wrapping `type_stream`

### Modified
- `src/main.py`
  - Removes `ResultThread` import and usage
  - `start_result_thread()` → `start_pipeline()`: creates `audio_q`, `text_q`, `recording_stopped` event, instantiates and starts the three workers
  - `stop_result_thread()` → `stop_pipeline()`: sets `stop_event` on RecorderWorker; sentinel cascade handles the rest
  - `on_transcription_complete()` removed — typing is now continuous, no single completion event
- `src/transcription.py`
  - `transcribe_local()` refactored to yield segments rather than return a joined string
  - Existing `transcribe()` (API path) unchanged — API mode remains batch
- `src/utils.py` / config schema
  - Add `hotwords: []` field under `model_options.local`

### Unchanged
- `src/input_simulation.py`
- `src/key_listener.py`
- All UI files

---

## Model Initialisation

GPU is preferred by default:

```python
device = config.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
```

If the user has explicitly set `device` in config, that value is used as-is. Otherwise: CUDA if available, CPU fallback. The existing `int8 → force CPU` rule is preserved.

---

## Status States

| State | Condition |
|-------|-----------|
| `recording` | RecorderWorker is running |
| `typing` | TyperWorker draining (RecorderWorker stopped) |
| `idle` | All workers done |

While recording and transcription overlap, status shows `recording` — the user cares about whether their mic is live, not internal pipeline state.

---

## Vocabulary / Hotwords

`config.yaml` gains:

```yaml
model_options:
  local:
    hotwords: []   # e.g. ["whisper-writer", "PyQt5", "xdotool"]
```

Passed directly to `faster_whisper.transcribe(hotwords=...)`.

---

## Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Burst shorter than `min_duration` | Discarded before hitting `audio_q` |
| User stops mid-burst | Partial burst flushed, sentinel cascades |
| Transcription error on a burst | Logged, burst skipped, pipeline continues |
| API mode selected | Unaffected — batch path unchanged |
| `text_q` grows (fast speech) | Queue drains in order; typing catches up naturally |

---

## What Does Not Change

- All recording modes (`hold_to_record`, `press_to_toggle`, `voice_activity_detection`, `continuous`)
- `InputSimulator` and all typing backends (pynput, xdotool, dotool, ydotool)
- Post-processing options (trailing period, capitalisation, trailing space)
- Settings UI
- API transcription path
