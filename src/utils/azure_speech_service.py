"""
Optional Azure Speech (Text-to-Speech) integration.

Env vars:
- AZURE_SPEECH_KEY
- AZURE_SPEECH_REGION

This module is optional. If the Azure Speech SDK isn't installed, functions
fail gracefully and return None/False.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def is_azure_speech_available() -> bool:
    """True if env vars are set AND the SDK imports successfully."""
    if not (os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION")):
        return False
    try:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore
        _ = speechsdk  # silence lint
        return True
    except Exception:
        logger.debug("Azure Speech SDK import failed", exc_info=True)
        return False


def synthesize_text_to_wav_bytes(
    text: str,
    voice: str = "en-US-JennyNeural",
) -> Optional[bytes]:
    """
    Synthesize *text* to WAV bytes using Azure Speech SDK.

    Prefers an **in-memory pull stream** to avoid temp-file I/O.
    Falls back to a temp file if the stream API is unavailable.

    Returns WAV bytes, or None on failure / unavailable.
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

        # ── Preferred: in-memory pull audio stream ────────────────────
        try:
            stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data  # bytes
            else:
                logger.warning(
                    "Speech synthesis (stream) did not complete: reason=%s",
                    result.reason,
                )
        except Exception:
            logger.debug("In-memory TTS stream failed; falling back to temp file", exc_info=True)

        # ── Fallback: temp file approach ──────────────────────────────
        import tempfile
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(text).get()

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.warning("Speech synthesis (file) did not complete: reason=%s", result.reason)
                return None

            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("Could not remove temp TTS file %s", tmp_path)

    except Exception:
        logger.warning("Azure Speech TTS failed", exc_info=True)
        return None
