"""
Optional Azure OpenAI integration for therapeutic explanations and recommendation text.
Uses env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION.
Load .env via python-dotenv in app if you use a .env file.
"""

import os
from typing import Optional

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
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        prompt = (
            f"In one or two short sentences, give a supportive, therapeutic explanation "
            f"for someone whose mood was detected as: {emotion_name} "
            f"(arousal={arousal:.2f}, valence={valence:.2f}). "
            f"Keep it warm and non-clinical; suggest why music can help."
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
        pass
    return None


def get_recommendation_blurb(
    emotion_quadrant: int,
    recommendations_list: list[str],
) -> Optional[str]:
    """
    Return a short AI blurb tying the recommendation list to the emotion.
    Uses Azure OpenAI if configured; otherwise returns None.
    """
    if not _is_configured() or not recommendations_list:
        return None
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        recs = ", ".join(recommendations_list[:5])
        prompt = (
            f"In one sentence, explain why these music styles might help right now: {recs}. "
            "Keep it encouraging and brief."
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
        pass
    return None


def is_azure_openai_available() -> bool:
    """Return True if Azure OpenAI env vars are set (for showing AI option in UI)."""
    return _is_configured()
