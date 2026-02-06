"""
Optional Azure Speech (Text-to-Speech) integration.

Env vars:
- AZURE_SPEECH_KEY
- AZURE_SPEECH_REGION

This module is optional. If the Azure Speech SDK isn't installed, functions
fail gracefully and return None/False.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


def is_azure_speech_available() -> bool:
    """True if env vars are set AND the SDK imports successfully."""
    if not (os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION")):
        return False
    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore
        _ = speechsdk  # silence lint
        return True
    except Exception:
        return False


def synthesize_text_to_wav_bytes(
    text: str,
    voice: str = "en-US-JennyNeural",
) -> Optional[bytes]:
    """
    Synthesize `text` to WAV bytes using Azure Speech SDK.
    Returns WAV bytes, or None on failure/unavailable.
    """
    if not is_azure_speech_available():
        return None
    if not text or not text.strip():
        return None

    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ["AZURE_SPEECH_KEY"],
            region=os.environ["AZURE_SPEECH_REGION"],
        )
        speech_config.speech_synthesis_voice_name = voice

        # Write to a temporary wav file, then read bytes back.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            return None

        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    except Exception:
        return None
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
