"""Create minimal synthetic WAV files for mel-spectrogram training demo."""
import os
import sys
import numpy as np
import soundfile as sf

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'audio')
SR = 22050
DURATION_SEC = 2.0  # Short clips

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for i in range(5):
        t = np.linspace(0, DURATION_SEC, int(SR * DURATION_SEC))
        # Vary frequency for different "moods"
        freq = 440 + i * 100
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.3
        path = os.path.join(AUDIO_DIR, f"sample_{i}.wav")
        sf.write(path, audio, SR)
        print(f"Wrote {path}")
    print(f"Created 5 synthetic WAV files in {AUDIO_DIR}")

if __name__ == "__main__":
    main()
