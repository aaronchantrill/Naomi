#!/usr/bin/env python3
import os
import requests
import tarfile
from tqdm import tqdm
from voice_assistant import app_utils

# List of files to download
files = [
    {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "name": "~/.local/share/Naomi/models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "extract": True
    },
    {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
        "name": "~/.local/share/Naomi/models/piper/en_US-amy-low.onnx",
        "extract": False
    },
    {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json",
        "name": "~/.local/share/Naomi/models/piper/en_US-amy-low.onnx.json",
        "extract": False
    },
    {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2",
        "name": "~/.local/share/Naomi/models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2",
        "extract": True
    }
]

if __name__ == "__main__":
    for file_info in files:
        if not os.path.isfile(file_info["name"]):
            app_utils.download_file(file_info["url"], file_info["name"])
        if file_info["extract"]:
            app_utils.extract_tar(file_info["name"])
    print("\nAll downloads and extractions complete!")
