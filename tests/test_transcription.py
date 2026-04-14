import numpy as np
from unittest.mock import MagicMock, patch
from transcription import transcribe_local_stream


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
