# AGENTS.md

## Cursor Cloud-specific instructions

### Project Overview

Emotion-Aware Music Therapy System — a Python/TensorFlow/Streamlit app that predicts arousal-valence emotions from audio and provides therapeutic music recommendations. Single-app monolith, no databases or external services required (Azure OpenAI is optional with hardcoded fallbacks).

### Running the App

- Create venv (first time only): `python -m venv .venv`
- Activate venv: `source .venv/bin/activate`
- Start Streamlit: `streamlit run src/frontend/app.py --server.port 8501 --server.headless true`
- The app has three input modes: Manual Emotion Selection, Audio Features Input, Audio Upload (A/V)

### Testing

- Run all tests: `pytest tests/ -v` (Azure calls are mocked)
- See `pytest.ini` for configuration

### Linting

- No project-specific linter config; `flake8 src/ --max-line-length=120` works but has pre-existing warnings

### Key Gotchas

- `python3.12-venv` system package must be installed before creating the virtualenv (`sudo apt-get install -y python3.12-venv`)
- TensorFlow install is large (~2GB); `pip install -r requirements.txt` can take 1-2 minutes
- Azure OpenAI env vars (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`) are optional; the app falls back to hardcoded recommendations when not set
- No `.env` file is needed to run tests or the Streamlit app; copy `.env.example` to `.env` only if Azure integration is desired
