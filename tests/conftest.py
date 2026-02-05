"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Skip TensorFlow tests if TensorFlow is not available or has DLL issues
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except (ImportError, OSError) as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)

# Skip audio tests if soundfile is not available
try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "tensorflow: tests that require TensorFlow")
    config.addinivalue_line("markers", "audio: tests that require audio processing libraries")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    skip_tensorflow = pytest.mark.skip(reason="TensorFlow not available or DLL error")
    skip_audio = pytest.mark.skip(reason="soundfile not available")
    
    for item in items:
        if "tensorflow" in item.keywords and not TF_AVAILABLE:
            item.add_marker(skip_tensorflow)
        if "audio" in item.keywords and not SOUNDFILE_AVAILABLE:
            item.add_marker(skip_audio)
