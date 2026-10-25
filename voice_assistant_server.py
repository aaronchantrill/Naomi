#!/usr/bin/env python3
# voice_assistant_server.py
#
# Thin entry point for the voice assistant package.
# Configuration is read from config.json in the same directory.
#
# Run with:
#   python voice_assistant_server.py
# or:
#   python -m voice_assistant

import asyncio
from voice_assistant.__main__ import start

if __name__ == "__main__":
    asyncio.run(start())
