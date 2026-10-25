# voice_assistant/llm.py
import asyncio
from fastapi import WebSocket
import json
from jinja2 import Template
import logging
import math
import os
from pathlib import Path
from pprint import pprint
import re
import requests
import aiohttp
from datetime import datetime
from .tts import TTSEngine


log = logging.getLogger(__name__)
re_remove_metadata = re.compile(r'^\s*\[USER:.*?\]\[DATETIME:.*?\]\s*')


# proxy for raise exception errors in jinja2 templates
def raise_exception(message):
    raise ValueError(message)


class LLMClient:
    def __init__(self, llm_cfg: dict, tts_cfg: dict) -> None:
        self.completion_url: str = llm_cfg.get("host", "")
        self.endpoint: str = llm_cfg.get("endpoint", "/v1/completions")
        self.max_tokens: int = llm_cfg.get("max_tokens", -1)
        self.temperature: float = llm_cfg.get("temperature", 1.0)
        self.api_key: str = llm_cfg.get("api_key", None)
        self.prompt_headers: dict = {'Authorization': self.api_key or "Insert your api key here"}
        self.enable_thinking = llm_cfg.get("enable_thinking", False)
        system_prompt_file: str = llm_cfg.get("system_prompt", "system_prompt.txt")
        with open(system_prompt_file, "r") as f:
            self.system_prompt: str = f.read()
        self.short_term_memory_log: str = llm_cfg.get("short_term_memory_log", "short_term_memories.json")
        self.long_term_memory_log: str = llm_cfg.get("long_term_memory_log", "long_term_memories.json")
        self.tts: TTSEngine = TTSEngine(tts_cfg=tts_cfg)
        # Get the LLM server properties
        props_url: str = f"{self.completion_url}/props"
        try:
            with requests.get(props_url, headers=self.prompt_headers) as response:
                data: dict = response.json()
                self.model_path: str = "unknown"
                if 'model_path' in data:
                    self.model_path = data['model_path']
                self.n_ctx: int = 2048
                if 'default_generation_settings' in data:
                    if 'n_ctx' in data['default_generation_settings']:
                        self.n_ctx = data['default_generation_settings']['n_ctx']
                # default to basic ChatML template
                self.chat_template: str = "".join([
                    "{{ bos_token }}",
                    "{% for message in messages %}",
                    "    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}",
                    "{% endfor %}",
                    "{% if add_generation_prompt %}",
                    "    {{ '<|im_start|>assistant\n' }}",
                    "{% endif %}"
                ]),
                if 'chat_template' in data:
                    self.chat_template = data['chat_template']
                # Check if a template file exists for the model
                log.info(f"Checking for template file {Path(self.model_path).name}.template")
                try:
                    with open(f"{os.path.join('templates', Path(self.model_path).name)}.template") as f:
                        self.chat_template = f.read()
                except Exception as e:
                    log.info(e, exc_info=True)
                print()
                print(self.model_path)
                print()
                print(self.chat_template)
                print()
                self.template: jinja2.environment.Template = Template(self.chat_template)
                self.template.globals['raise_exception'] = raise_exception
                self.bos_token: str = '<|endoftext|>'
                if 'bos_token' in data:
                    self.bos_token = data['bos_token']
                self.eos_token: str = '<|im_end|>'
                if 'eos_token' in data:
                    self.eos_token = data['eos_token']
        except requests.exceptions.ConnectionError as e:
            log.error(e)
            exit(1)
        self.short_term_memories: list[dict] = []
        try:
            with open(self.short_term_memory_log, "r") as f:
                self.short_term_memories = json.load(f)
        except FileNotFoundError:
            now = datetime.now()
            self.short_term_memories = [
                {
                    "name": "system",
                    "is_user": False,
                    "is_system": True,
                    "send_date": f"{now.strftime('%B %d, %Y %I:%M%p').lower()}",
                    "mes": self.system_prompt,
                    "extra":{
                        "completion_url": f"{self.completion_url}{self.endpoint}",
                        "model_path": self.model_path
                    }
                }
            ]
        try:
            with open(self.long_term_memory_log, "r") as f:
                self.long_term_memories = json.load(f)
        except FileNotFoundError:
            self.long_term_memories = []
        # Check if the current messages fit in the context window.
        prompt = self.template.render(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]+[
                {
                    "role": "user" if message["is_user"] else message["name"],
                    "content": f"[USER: {message['name']}][DATETIME: {message['send_date']}] {message['mes']}"
                } for message in self.short_term_memories
            ],
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=True
        )
        token_count: int = 0
        with requests.post(
            f"{self.completion_url}/tokenize",
            json={
                "content": prompt,
                "add_bos": False,
                "special": True
            }
        ) as response:
            data: dict = response.json()
            token_count = len(data['tokens'])
        log.info(f"Tokens: {token_count} of {self.n_ctx} tokens")
        while token_count > self.n_ctx * 0.9:
            self.update_long_term_memory(token_count)
            prompt = self.template.render(
                messages=[{"role": "user" if message["is_user"] else message["name"], "content": f"[USER: {message['name']}][DATETIME: {message['send_date']}] {message['mes']}"} for message in self.short_term_memories],
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                add_generation_prompt=True
            )
            token_count: int = 0
            with requests.post(
                f"{self.completion_url}/tokenize",
                json={
                    "content": prompt,
                    "add_bos": False,
                    "special": True
                }
            ) as response:
                data: dict = response.json()
                token_count = len(data['tokens'])
            log.info(f"Tokens: {token_count} of {self.n_ctx} tokens")

    async def handle(self, user: str, message: str, websocket: WebSocket) -> None:
        self.append_message(
            role=user,
            is_user=True,
            is_system=False,
            content=message
        )
        prompt: str = self.template.render(
            messages=[{"role": "user" if message["is_user"] else message["name"], "content": f"[USER: {message['name']}][DATETIME: {message['send_date']}] {message['mes']}"} for message in self.short_term_memories],
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=True,
            enable_thinking = self.enable_thinking
        )
        print((prompt.strip())[-7:])
        # prompt = f"Question: {message}\nAnswer:"
        # pprint(self.short_term_memories)
        # Send the POST request to the API server
        sentence: list = []
        sentences: list = []
        escape_count: int = 0
        update_long_term_memory: bool = False
        total_tokens: int = 0
        try:
            with requests.post(
                f"{self.completion_url}{self.endpoint}",
                json={
                    "model": self.model_path,
                    "stream": True,
                    "max_tokens": -1,
                    "user": "Aaron",
                    "prompt": prompt
                },
                stream=True
            ) as response:
                # If the model has been instructed to think, then wait for a </think> token
                thinking = False
                # Qwen 3.8 adds a "<think>" tag at the end if add_generation_prompt is true and enable_thinking is also true.
                if (prompt.strip())[-7:] == "<think>":
                    thinking = True
                for line in response.iter_lines():
                    # print(line)
                    if line:
                        info: dict = self._clean_raw_bytes(line)
                        # usage.total_tokens should only appear in the last line
                        if 'usage' in info:
                            if 'total_tokens' in info['usage']:
                                total_tokens = info['usage']['total_tokens']
                                print(f"total_tokens = {total_tokens}")
                                if total_tokens / self.n_ctx > 0.9:
                                    print("Updating long term memory")
                                    print(f"{total_tokens}/{self.n_ctx} tokens used")
                                    update_long_term_memory = True
                        next_token = self._process_line(info)
                        # Qwen uses <think>...</think> tags to show thoughts
                        if next_token == "<think>":
                            thinking = True
                        if next_token == "</think>":
                            thinking = False
                            continue
                        # Gemma 4 uses <|channel>...<channel|> tags to delimit thoughts.
                        if next_token == "<|channel>":
                            thinking = True
                        if next_token == "<channel|>":
                            thinking = False
                            continue
                        if next_token:
                            print(f"\x1b[36m*{next_token}* \x1b[0m", end='')
                            if next_token == self.eos_token:
                                print()
                                break
                            # For some reason the rocket llm goes into an infinite loop of *E* sometimes.
                            # Is it screaming?
                            if next_token == 'E':
                                escape_count += 1
                                if escape_count == 3:
                                    print()
                                    sentence = []
                                    break
                            else:
                                escape_count = 0
                            if not thinking:
                                sentence.append(next_token)
                                # Check if this is the end of a sentence
                                if next_token in ['.', '?', '!']:
                                    # Send text to front end
                                    print()
                                    log.info(f'Sending partial LLM result, token = {next_token}')
                                    # Since we added meta information above, the LLM will probably
                                    # want to send meta information back.
                                    current_sentence = re_remove_metadata.sub(r"", "".join(sentence))
                                    try:
                                        await websocket.send_text(json.dumps({"type": "tts_text", "detail": current_sentence}))
                                    except Exception as e:
                                        raise e

                                    audio_bytes = await self.tts.synthesize(current_sentence)
                                    try:
                                        await websocket.send_bytes(audio_bytes)
                                        await websocket.send_text(json.dumps({"type": "tts_done"}))
                                        log.info("TTS synthesis complete.")
                                    except Exception as e:
                                        raise e
                                    sentences.append(current_sentence)
                                    sentence = []
                if len(sentence):
                    # Send text to front end
                    log.info("Sending final LLM result")
                    current_sentence = re_remove_metadata.sub(r"", "".join(sentence))
                    print()
                    try:
                        await websocket.send_text(json.dumps({"type": "tts_text", "detail": current_sentence}))
                    except Exception as e:
                        raise e

                    audio_bytes = await self.tts.synthesize(current_sentence)
                    try:
                        await websocket.send_bytes(audio_bytes)
                        await websocket.send_text(json.dumps({"type": "tts_done"}))
                        log.info("TTS synthesis complete.")
                    except Exception as e:
                        raise e
                    sentences.append(current_sentence)
                    sentence = []
                else:
                    log.info("Final LLM result was empty")
        except Exception as e:
            log.error(e, exc_info=True)
            current_sentence = f"There is a problem with my AI: {e}"
            sentences.append(current_sentence)
            audio_bytes = await self.tts.synthesize(current_sentence)
            try:
                await websocket.send_bytes(audio_bytes)
                await websocket.send_text(json.dumps({"type": "tts_done"}))
                log.info("TTS synthesis complete.")
            except Exception as e:
                raise e

        content = "".join(sentences)
        if len(content):
            self.append_message(role="assistant", is_user=False, is_system=False, content=content)
        else:
            # Send something if the LLM's response is completely empty
            current_sentence = "Filtered"
            try:
                await websocket.send_text(json.dumps({"type": "tts_text", "detail": current_sentence}))
            except Exception as e:
                raise e

            audio_bytes = await self.tts.synthesize(current_sentence)
            try:
                await websocket.send_bytes(audio_bytes)
                await websocket.send_text(json.dumps({"type": "tts_done"}))
                log.info("TTS synthesis complete.")
            except Exception as e:
                raise e
            self.append_message(role="assistant", is_user=False, is_system=False, content=content)
        # Check to see if the current context is getting too large
        if update_long_term_memory:
            self.update_long_term_memory(total_tokens)
        # Save the conversation after each round.
        self.save_conversation()

    def append_message(self, role, is_user, is_system, content):
        """Append a message to both internal _messages list and the log file"""
        # {"name":"Talkie Toaster","is_user":false,"is_system":false,"send_date":"June 21, 2025 2:05pm","mes":"\"Howdy, Doodly Doo. I'm Talkie, Talkie Toaster, your chirpy, breakfast companion. Talkie's the name, toasting's the game. Anybody want any toast?\"","extra":{}}
        # {"name":"Drask","is_user":true,"is_system":false,"send_date":"June 21, 2025 2:05pm","mes":"\"No thanks, Talkie. Can you tell me what's going on in the news?\"","extra":{"isSmallSys":false,"reasoning":""},"force_avatar":"User Avatars/user-default.png"}
        now = datetime.now()
        self.short_term_memories.append(
            {
                'name': role,
                'is_user': is_user,
                'is_system': is_system,
                'mes': content,
                'send_date': f"{now.strftime('%B %d, %Y %I:%M')}{now.strftime('%p').lower()}",
                'extra': {
                    'completion_url': self.completion_url,
                    'model_path': self.model_path
                }
            }
        )

    def _clean_raw_bytes(self, line: str) -> dict:
        line = line.decode("utf-8")
        if line:
            line = line.removeprefix("data: ")
            # print(line)
            # print(f"Line: {line}")
            if line == '[DONE]':
                # line = '{"choices": [{"text": "' + self.eos_token + '"}]}'
                line = '{"content": "' + self.eos_token +'"}'
                # print(f"Line: {line}")
            line = json.loads(line)
            return line
        else:
            return {}

    def _process_line(self, line: dict) -> str:
        token: str = self.eos_token
        if 'error' in line:
            print(line['error'])
        else:
            if not (('stop' in line and line['stop']) or ('choices' in line and 'finish_reason' in line['choices'][0] and line['choices'][0]['finish_reason'] == 'stop')):
                if 'choices' in line:
                    token = line['choices'][0]['text']
                elif 'content' in line:
                    token = line['content']
                else:
                    log.warn("Unknown structure returned from LLM")
        return token

    def update_long_term_memory(self, total_tokens: int) -> None:
        """
        Ask the LLM to summarize the conversation up to this point. This
        becomes the second message in short term memory, after the system
        prompt. Me
        """
        summary: str = ""
        # archive messages approximately half the size of the context window
        # Start by getting the approximate number of tokens per message.
        tokens_per_text: int = total_tokens / len(self.short_term_memories)
        # Next, figure out how many messages fit in 1/2 n_ctx
        number_of_memories_to_archive = int((self.n_ctx * 0.5) // tokens_per_text)
        print(f"Archiving {number_of_memories_to_archive} messages")
        memories_to_archive: list[dict] = self.short_term_memories[:number_of_memories_to_archive]
        memories_to_archive.append({
            'name': 'user',
            'is_user': True,
            'is_system': False,
            'mes': 'Please summarize the conversation so far for yourself using as many words as you need.'
        })
        prompt = self.template.render(
            messages=[
                {
                    "role": "user" if message["is_user"] else message["name"],
                    "content": message['mes']
                } for message in memories_to_archive
            ],
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_generation_prompt=True
        )
        data = {
            "stream": True,
            "prompt": prompt
        }
        try:
            with requests.post(
                f"{self.completion_url}{self.endpoint}",
                headers=self.prompt_headers,
                json=data,
                stream=True
            ) as response:
                for line in response.iter_lines():
                    if line:
                        info: dict = self._clean_raw_bytes(line)
                        next_token = self._process_line(info)
                        if next_token:
                            print(f"\x1b[36m*{next_token}* \x1b[0m", end='')
                            if next_token == self.eos_token:
                                print()
                                break
                            summary += next_token
        except requests.exceptions.ConnectionError:
            log.warn(f"Error connecting to {self.completion_url}")
        # Add the memories to long term memory (skip the system prompt and instruction to summarize the conversation)
        self.long_term_memories.extend([message for message in memories_to_archive if message['name'] != 'system'])
        # Now remove the memories from short term memory and restore the system prompt and summary
        now = datetime.now()
        short_term_memories = [
            {
                "name": "assistant",
                "is_user": False,
                "is_system": False,
                "send_date": f"{now.strftime('%B %d, %Y %I:%M')}{now.strftime('%p').lower()}",
                "mes": summary,
                "extra": {
                    "completion_url": f"{self.completion_url}{self.endpoint}",
                    "model_path": self.model_path
                }
            }
        ]
        short_term_memories.extend(self.short_term_memories[number_of_memories_to_archive:])
        self.short_term_memories = short_term_memories
        self.save_conversation()

    def save_conversation(self) -> None:
        """
        Writes both short and long term memories to text files
        """
        log.info("Saving short term memories")
        with open(self.short_term_memory_log, "w") as f:
            json.dump(self.short_term_memories, f, indent=4)
        log.info("Saving long term memories")
        with open(self.long_term_memory_log, "w") as f:
            json.dump(self.long_term_memories, f, indent=4)
        log.info("Memories saved")
