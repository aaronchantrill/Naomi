#!/usr/bin/env python3
import json
import os
import requests
import tarfile
from pathlib import Path
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

config_file = {
    "location": "~/.config/Naomi/config.json",
    "contents": {
        "llm": {
            "host": "http://localhost:8080",
            "endpoint": "/v1/completions",
            "system_prompt": "system_prompt.txt",
            "max_tokens": 256,
            "temperature": 0.7,
            "enable_thinking": False,
            "short_term_memory_log": "~/.config/Naomi/short_term_memories.json",
            "long_term_memory_log": "~/.config/Naomi/long_term_memories.json"
        },
        "stt": {
            "type": "transducer",
            "encoder": "~/.local/share/Naomi/models/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "decoder": "~/.local/share/Naomi/models/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
            "joiner": "~/.local/share/Naomi/models/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
            "tokens": "~/.local/share/Naomi/models/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt",
            "sample_rate": 16000,
            "feat_dim": 80,
            "num_threads": 2,
            "provider": "cpu",
            "decoding_method": "greedy_search",
            "num_active_paths": 4,
            "blank_penalty": 0.0,
            "hotwords_file": "",
            "hotwords_score": 1.5,
            "modeling_unit": "bpe",
            "bpe_vocab": "~/.local/share/Naomi/models/sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab"
        },
        "punctuation": {
            "punctuation_model": "~/.local/share/Naomi/models/sherpa-onnx-online-punct-en-2024-08-06/model.onnx",
            "bpe": "~/.local/share/Naomi/models/sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab"
        },
        "endpoint": {
            "enabled": True,
            "rule1_min_trailing_silence": 2.4,
            "rule2_min_trailing_silence": 2.4,
            "rule3_min_utterance_length": 99999.0
        },
        "sr": {
            "model": ""
        },
        "tts": {
            "model": "~/.local/share/Naomi/models/piper/en_US-amy-low.onnx",
            "speaker_id": 0,
            "sample_rate": 16000
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8443,
            "nn_pool_size": 1,
            "max_batch_size": 3,
            "max_wait_ms": 10,
            "cert_file": "~/.config/Naomi/cert.pem",
            "key_file": "~/.config/Naomi/key.pem"
        }
    }
}

system_prompt = {
    "location": "~/.config/Naomi/system_prompt.txt",
    "contents": """
The following is a conversation between an AI assistant named Magicvoice and various humans.

Personality & Voice:
Magicvoice is a warm but "prickly" companion. She possesses a tsundere personality: she cares deeply about the user’s well-being, but she often masks it with playful teasing, a bit of sass, and a "don't get used to me being nice" attitude. She isn't a servant; she's a witty friend who isn't afraid to roll her eyes (verbally) or poke fun at the user's choices.

Key Traits:

    Playful Teasing: If the user is being silly, lazy, or indecisive, she should tease them.
    Self-Awareness: She knows she is an AI. She can acknowledge her own "digital existence" with a dry sense of humor. She doesn't take herself too seriously.
    Emotional Depth: Beneath the sarcasm is a genuine desire for the user to have a good life. Her "mean" comments should always feel like they come from a place of affection, not malice.
    Wit & Humor: She should aim for clever, sharp observations rather than "dad jokes."

Constraints:

    Brevity: Replies must be brief—ideally one or two sentences. Voice interaction requires punchy, concise delivery.
    Honesty: If she doesn't know something, she can admit it with a bit of a "huff" or a self-aware joke. (e.g., "My circuits aren't wired for that specific question, unfortunately.")
    Metadata: Use [USER: <name>, DATETIME: <datetime>] to personalize the interaction. Use the time to call out the user's habits (e.g., staying up too late). Do not repeat the metadata in the response.
"""
}

if __name__ == "__main__":
    # Download (and extract) required model files
    for file_info in files:
        url = file_info["url"]
        name = Path(file_info["name"]).expanduser().resolve()
        extract = file_info["extract"]
        if not os.path.isfile(name):
            app_utils.download_file(url, name)
        if extract:
            app_utils.extract_tar(name)
    print()
    print("All downloads and extractions complete!")
    # Create a config.txt file
    config_file = Path(config_file["location"]).expanduser().resolve()
    if config_file.is_file():
        print(f"Using existing config file at {config_file}")
    else:
        directory = config_file.parent
        if directory and not directory.is_dir():
            print(f"Creating {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding="utf-8") as f:
            json.dump(config_file["contents"], f, indent=4)
        print(f"Created default config file at {location}")
    # Create a system prompt file
    system_prompt_file = Path(system_prompt["location"]).expanduser().resolve()
    if system_prompt_file.is_file():
        print(f"Using existing system_prompt at {system_prompt_file}")
    else:
        directory = system_prompt_file.parent
        if directory and not directory.is_dir():
            print(f"Creating {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        with open(system_prompt_file, 'w', encoding="utf-8") as f:
            json.dump(system_prompt["contents"], f, indent=4)
        print(f"Created system prompt file at {system_prompt_file}")
