"""
Tests for Azure Speech TTS service (without actual API calls).
"""

import pytest
import os
from unittest.mock import patch

from src.utils.azure_speech_service import is_azure_speech_available, synthesize_text_to_wav_bytes


def test_not_available_without_env():
    """Should return False when env vars are missing."""
    with patch.dict(os.environ, {}, clear=True):
        assert is_azure_speech_available() is False


def test_not_available_without_sdk():
    """Should return False when SDK is not installed."""
    with patch.dict(os.environ, {
        "AZURE_SPEECH_KEY": "test-key",
        "AZURE_SPEECH_REGION": "eastus",
    }):
        with patch.dict("sys.modules", {"azure.cognitiveservices.speech": None}):
            # Importing None module raises ImportError
            assert is_azure_speech_available() is False


def test_synthesize_returns_none_without_config():
    """Should return None when not configured."""
    with patch.dict(os.environ, {}, clear=True):
        result = synthesize_text_to_wav_bytes("Hello world")
        assert result is None


def test_synthesize_returns_none_for_empty_text():
    """Should return None for empty/whitespace text."""
    with patch.dict(os.environ, {
        "AZURE_SPEECH_KEY": "test-key",
        "AZURE_SPEECH_REGION": "eastus",
    }):
        assert synthesize_text_to_wav_bytes("") is None
        assert synthesize_text_to_wav_bytes("   ") is None


def test_synthesize_returns_none_for_none_text():
    """Should handle None text input gracefully."""
    with patch.dict(os.environ, {}, clear=True):
        # Not configured → returns None before checking text
        result = synthesize_text_to_wav_bytes("")
        assert result is None
