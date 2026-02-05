"""
Tests for Azure OpenAI integration (without actual API calls).
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Import the module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.azure_openai_service import (
    _is_configured,
    _get_client,
    get_therapeutic_explanation,
    get_recommendation_blurb,
    get_ai_music_styles,
    get_ai_sample_tracks,
    is_azure_openai_available,
)


def test_is_configured_without_env():
    """Test _is_configured returns False when env vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        assert _is_configured() is False


def test_is_configured_with_env():
    """Test _is_configured returns True when env vars are set."""
    with patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }):
        assert _is_configured() is True


def test_get_client_raises_without_config():
    """Test _get_client raises when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            _get_client()


def test_get_therapeutic_explanation_without_config():
    """Test get_therapeutic_explanation returns None when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = get_therapeutic_explanation(0, 0.5, 0.3, "Sad")
        assert result is None


def test_get_therapeutic_explanation_with_mock_api():
    """Test get_therapeutic_explanation with mocked API call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Music can help soothe your emotions."

    with patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
    }):
        with patch('src.utils.azure_openai_service._get_client') as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            result = get_therapeutic_explanation(0, 0.5, 0.3, "Sad")
            assert result == "Music can help soothe your emotions."
            assert mock_client.chat.completions.create.called


def test_get_ai_music_styles_without_config():
    """Test get_ai_music_styles returns None when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = get_ai_music_styles(0, 0.5, 0.3, "Sad")
        assert result is None


def test_get_ai_music_styles_with_mock_api():
    """Test get_ai_music_styles with mocked API call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Slow acoustic folk\nGentle piano ballads\nAmbient soundscapes"

    with patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }):
        with patch('src.utils.azure_openai_service._get_client') as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            result = get_ai_music_styles(0, 0.5, 0.3, "Sad")
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0
            assert "Slow acoustic folk" in result or any("acoustic" in s.lower() for s in result)


def test_get_ai_sample_tracks_without_config():
    """Test get_ai_sample_tracks returns None when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = get_ai_sample_tracks(0, 0.5, 0.3, "Sad")
        assert result is None


def test_get_ai_sample_tracks_with_mock_api():
    """Test get_ai_sample_tracks with mocked API call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Peaceful Rain | Nature Sounds | 5:00\nGentle Waves | Ocean Ambience | 4:30"

    with patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }):
        with patch('src.utils.azure_openai_service._get_client') as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_client_fn.return_value = mock_client

            result = get_ai_sample_tracks(0, 0.5, 0.3, "Sad")
            assert result is not None
            assert isinstance(result, list)
            assert len(result) > 0
            assert "title" in result[0]
            assert "artist" in result[0]
            assert "duration" in result[0]


def test_get_recommendation_blurb_without_config():
    """Test get_recommendation_blurb returns None when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = get_recommendation_blurb(0, ["Jazz", "Blues"])
        assert result is None


def test_is_azure_openai_available():
    """Test is_azure_openai_available matches _is_configured."""
    with patch.dict(os.environ, {}, clear=True):
        assert is_azure_openai_available() is False

    with patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
    }):
        assert is_azure_openai_available() is True
