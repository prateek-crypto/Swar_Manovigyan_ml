"""
Optional Azure OpenAI integration for therapeutic explanations and **AI-driven music suggestions**.

Uses env:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_DEPLOYMENT
- AZURE_OPENAI_API_VERSION

Load .env via python-dotenv in app if you use a .env file.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── Default duration for AI-generated sample tracks when parsing fails ────
_DEFAULT_TRACK_DURATION = "3:30"

# Try optional dotenv so .env is loaded when present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _is_configured() -> bool:
    return bool(
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        and os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    )


def _get_client():
    """Create an AzureOpenAI client, or raise if misconfigured."""
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def get_therapeutic_explanation(
    emotion_quadrant: int,
    arousal: float,
    valence: float,
    emotion_name: str,
) -> Optional[str]:
    """
    Return a short therapeutic explanation for the detected emotion quadrant.
    Uses Azure OpenAI if configured; otherwise returns None.
    """
    if not _is_configured():
        return None
    try:
        client = _get_client()
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        prompt = (
            f"In one or two short sentences, give a supportive, therapeutic explanation "
            f"for someone whose mood was detected as: {emotion_name} "
            f"(arousal={arousal:.2f}, valence={valence:.2f}). "
            f"Keep it warm, non-clinical, culturally neutral, and avoid medical claims."
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        if resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    except Exception:
        logger.warning("Azure OpenAI therapeutic explanation failed", exc_info=True)
        return None
    return None


def get_recommendation_blurb(
    emotion_quadrant: int,
    recommendations_list: List[str],
) -> Optional[str]:
    """
    Return a short AI blurb tying the recommendation list to the emotion.
    Uses Azure OpenAI if configured; otherwise returns None.
    """
    if not _is_configured() or not recommendations_list:
        return None
    try:
        client = _get_client()
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        recs = ", ".join(recommendations_list[:5])
        prompt = (
            f"In one sentence, explain why these music styles might help right now: {recs}. "
            "Keep it encouraging, gentle, and avoid medical language."
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.6,
        )
        if resp.choices and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    except Exception:
        logger.warning("Azure OpenAI recommendation blurb failed", exc_info=True)
        return None
    return None


def get_ai_music_styles(
    emotion_quadrant: int,
    arousal: float,
    valence: float,
    emotion_name: str,
    max_items: int = 7,
) -> Optional[List[str]]:
    """
    Use Azure OpenAI to generate high-level music style/playlist suggestions
    tailored to the detected emotion and A/V coordinates.

    Returns a list of short style/playlist descriptions, or None on failure.
    """
    if not _is_configured():
        return None
    try:
        client = _get_client()
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        prompt = (
            f"You are a music therapist helping someone who currently feels: {emotion_name} "
            f"with arousal={arousal:.2f} and valence={valence:.2f} in a 0-1 space.\n"
            f"Suggest up to {max_items} concise music styles or playlist descriptions that could "
            f"support emotion regulation in this state. Focus on genres, tempos, and textures "
            f"(e.g., 'slow acoustic folk with soft vocals', 'gentle lo-fi beats for winding down').\n"
            "Therapeutic guidelines:\n"
            "- Avoid explicit medical claims or diagnoses.\n"
            "- Consider both mood-matching (meeting them where they are) and gentle mood-shifting.\n"
            "- Be culturally neutral and language-agnostic.\n"
            "Output format:\n"
            "- Return each suggestion on its own line.\n"
            "- Do not include numbering, bullets, or extra commentary."
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.8,
        )
        if not resp.choices or not resp.choices[0].message.content:
            return None
        content = resp.choices[0].message.content
        lines = [ln.strip(" -•\t") for ln in content.splitlines() if ln.strip()]
        suggestions = [ln for ln in lines if ln]
        return suggestions[:max_items] if suggestions else None
    except Exception:
        logger.warning("Azure OpenAI music styles generation failed", exc_info=True)
        return None


def get_ai_sample_tracks(
    emotion_quadrant: int,
    arousal: float,
    valence: float,
    emotion_name: str,
    max_items: int = 5,
) -> Optional[List[Dict[str, str]]]:
    """
    Use Azure OpenAI to suggest example track-like entries (title/artist/duration).

    **Note:** These are AI-generated illustrative suggestions — not guaranteed to
    correspond to real licensed songs.  The UI should make this clear to the user.

    Returns list of dicts: {"title", "artist", "duration"}, or None on failure.
    """
    if not _is_configured():
        return None
    try:
        client = _get_client()
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        prompt = (
            f"Suggest {max_items} real, well-known tracks for someone who feels {emotion_name} "
            f"(arousal={arousal:.2f}, valence={valence:.2f}). "
            "Pick songs that are widely available on streaming platforms.\n"
            "For each track, include:\n"
            "- the real title\n"
            "- the real artist name\n"
            "- an approximate duration in mm:ss\n"
            "Output format:\n"
            "- Return each track on its own line, in the format: Title | Artist | mm:ss\n"
            "- Do not use bullets, numbering, or additional commentary."
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=260,
            temperature=0.9,
        )
        if not resp.choices or not resp.choices[0].message.content:
            return None
        content = resp.choices[0].message.content
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        tracks: List[Dict[str, str]] = []
        for ln in lines:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 2:
                title = parts[0]
                artist = parts[1]
                duration = parts[2] if len(parts) >= 3 else _DEFAULT_TRACK_DURATION
                tracks.append({"title": title, "artist": artist, "duration": duration})
            if len(tracks) >= max_items:
                break
        return tracks or None
    except Exception:
        logger.warning("Azure OpenAI sample tracks generation failed", exc_info=True)
        return None


def is_azure_openai_available() -> bool:
    """Return True if Azure OpenAI env vars are set (for showing AI option in UI)."""
    return _is_configured()
