# Bringing Naomi to your Pocket

This is the first pull request in a new branch for what I hope will be a compelling new direction for Naomi development.

The primary goals are to move beyond the "smart speaker" model in which the assistant is tied to a specific piece of hardware to an HTML based interface that can run on any web browser, including smart phones and other mobile devices, and to replace keyword and phrase based intent parsing with an LLM.

## Install

Naomi is installed from source code and written in mostly Python. You will need an OpenAI compatible http /v1/completions API to connect to, which is offered by most LLM providers, whether cloud based or local. I am specifically using Llama.cpp llama-server running on port 8080 and hosting various models while developing this branch of Naomi.

1. git clone this repo
    ```bash
    git clone -b NaomiLLM https://github.com/NaomiProject/Naomi.git
    ```
2. cd into the repo
    ```bash
    cd Naomi
    ```
3. create and activate a virtual environment
    ```bash
    python3 -m venv .venv --prompt Naomi
    source .venv/bin/activate
    ```
4. Install required packages
    ```bash
    pip3 install -r requirements.txt
    ```
5. Download the models (asr, tts, punctuation)
    ```bash
    python3 download_models.py
    ```
6. Edit the "llm" configuration file to point the llm host at your chosen LLM (unless you are using localhost port 8080).
7. Run Naomi.py
    ```bash
    python3 Naomi.py
    ```

## Key Features in this commit include

* Web Interface: A simple, encrypted HTML front-end that can be accessed from a web browser running on a smart phone.
* Automatic TLS certificate generation.
* Ability to chat with your LLM through either ASR or directly by typing into the interface.
* Support for OpenAI compatible APIs including local LLM servers such as Llama.cpp, Ollama, vLLM, Unsloth, etc.
* Short and Long term memories - as the llm runs out of space in the context window, memories are automatically shifted from long to short term memories. All memories are stored in easy to read and manipulate JSON documents.

## Upcoming projects

* Restore the Naomi plugin system. The Naomi plugin system made creating plugins easy for developers.
* Access to long term memories.
* Speaker recognition.
* Remote access to your assistant via email and text messages.
* Access to MCP tools - basically the old speechhandler plugins (skills) will be implemented through tools the LLM will be able to call.
* Ability to follow multiple conversations through different devices without getting confused.
* Addition of interfaces that can be used to adjust system settings through the front-end.
* Python front-end that will allow Naomi to work like a smart speaker on Raspberry Pi.

## Caveats
* While long term memories are stored, there is no way to access them at this moment. The goal is to start by adding RAG access through FAISS and eventually move to an LLM Wiki, possibly using Open Knowledge Format document structure.
* All conversation threads currently go to the same context history document regardless of what interface is being used to converse with the LLM. This can lead to the LLM becoming confused if being spoken to through more than one interface at the same time.
* I expect speechhandler skills to be reimplemented as MCP tools that the LLM can access as needed. Other plugin types such as speech to text, text to speech, speaker recognition and vad will be implemented using the old Naomi plugin system.
* Naomi can't really run comfortably on a Raspberry Pi anymore, which was one of the original goals for the Naomi project. While the original Naomi project aimed for Raspberry Pi compatibility, the hardware requirements for modern artificial intelligence have evolved significantly. While a high-spec Raspberry Pi 5 could still handle much of the load, it's often more cost effective to use a Linux desktop for the main processing.
* I do all my development on Debian Linux, and so some assumptions about running on a Linux operating system may be implicit in my implementations. I have nothing against running Naomi on other operating systems, and Naomi is still implemented in pure Python so it should run on any system that Python can be installed on. If you are running Naomi on another operating system and find that some adjustments are necessary to get it working, please make a pull request with your changes.

## Acknowledgments:

I’d like to acknowledge the inspiration and approaches I was able to learn while reading the source of and working with the [GLaDOS](https://github.com/dnhkng/GLaDOS), [Rhasspy](https://github.com/rhasspy/rhasspy) and [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) projects in bringing LLM support and an HTML front end to the Naomi project.
