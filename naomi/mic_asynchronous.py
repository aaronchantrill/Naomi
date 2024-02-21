import collections
import contextlib
import logging
import random
import tempfile
import threading
import time
import wave
from naomi import alteration
from naomi import i18n
from naomi import paths
from naomi import profile
from naomi import visualizations
from datetime import datetime


recordings_available_event = threading.Event()


class Unexpected(Exception):
    def __init__(
        self,
        utterance
    ):
        self.utterance = utterance


class Mic(i18n.GettextMixin):
    actions_thread = None
    Continue = True

    def __init__(self, *args, **kwargs):
        translations = i18n.parse_translations(paths.data('locale'))
        i18n.GettextMixin.__init__(self, translations, profile)
        self.keywords = kwargs['keywords']
        self._input_device = kwargs['input_device']
        self._output_device = kwargs['output_device']
        self.passive_stt_plugin = kwargs['passive_stt_plugin']
        self.active_stt_plugin = kwargs['active_stt_plugin']
        self.special_stt_slug = kwargs['special_stt_slug']
        self.vad_plugin = kwargs['vad_plugin']
        self.tts_engine = kwargs['tts_engine']
        self.brain = kwargs['brain']
        self.recordings_queue = collections.deque([], maxlen=10)
        self.actions_queue = collections.deque([], maxlen=10)
        self._logger = logging.getLogger(__name__)


    def queue_recording(self, audio):
        self.recordings_queue.appendleft(audio)
        recordings_available_event.set()

    def queue_action(self, action):
        self.actions_queue.appendleft(action)

    @contextlib.contextmanager
    def _write_frames_to_file(self, frames, volume):
        """This is used internally"""
        with tempfile.NamedTemporaryFile(
            mode='w+b',
            suffix=".wav",
            prefix=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ) as f:
            wav_fp = wave.open(f, 'wb')
            wav_fp.setnchannels(self._input_device._input_channels)
            wav_fp.setsampwidth(int(self._input_device._input_bits / 8))
            wav_fp.setframerate(self._input_device._input_rate)
            fragment = b''.join(frames)
            if volume is not None:
                maxvolume = audioop.minmax(
                    fragment,
                    self._input_device._input_bits / 8
                )[1]
                fragment = audioop.mul(
                    fragment,
                    int(self._input_device._input_bits / 8),
                    volume * (2. ** 15) / maxvolume
                )

            wav_fp.writeframes(fragment)
            wav_fp.close()
            f.seek(0)
            yield f

    @contextlib.contextmanager
    def special_mode(self, name, phrases):
        plugin_info = profile.get_arg('plugins').get_plugin(
            self.special_stt_slug,
            category='stt'
        )
        plugin_config = profile.get_profile()

        original_stt_plugin = self.active_stt_plugin

        # If the special_mode engine is not specifically set,
        # copy the settings from the active stt engine.
        try:
            mode_stt_engine = plugin_info.plugin_class(
                name,
                phrases,
                plugin_info,
                plugin_config
            )
            if(profile.check_profile_var_exists(['special_stt'])):
                if(profile.check_profile_var_exists([
                    'special_stt',
                    'samplerate'
                ])):
                    mode_stt_engine._samplerate = int(
                        profile.get_profile_var([
                            'special_stt',
                            'samplerate'
                        ])
                    )
                if(profile.check_profile_var_exists([
                    'special_stt',
                    'volume_normalization'
                ])):
                    mode_stt_engine._volume_normalization = float(
                        profile.get_profile_var([
                            'special_stt',
                            'volume_normalization'
                        ])
                    )
            else:
                mode_stt_engine._samplerate = original_stt_engine._samplerate
                mode_stt_engine._volume_normalization = original_stt_engine._volume_normalization
            self.active_stt_plugin = mode_stt_engine
            yield
        finally:
            self.active_stt_plugin = original_stt_plugin

    def check_for_keyword(self, phrase, keywords=None):
        if not keywords:
            keywords = self.keywords
        # This allows multi-word keywords like 'Hey Naomi' or 'You there'
        wakewords = []
        for word in keywords:
            for w in phrase:
                if word.upper() in w.upper():
                    wakewords.append(word.upper())
        if any(wakewords):
            return True
        return False

    def listen(self):
        """Grab the next block of audio out of the queue and convert to text"""
        transcription = ""
        recordings_available_event.wait()
        try:
            audio = self.recordings_queue.pop()
            if len(audio)>0:
                with self._write_frames_to_file(audio, None) as f:
                    passive_transcription = self.passive_stt_plugin.transcribe(f)
                    if len(passive_transcription) > 0:
                        visualizations.run_visualization(
                            "output",
                            f"<  {passive_transcription}"
                        )
                        if(self.check_for_keyword(passive_transcription)):
                            active_transcription = [" ".join(self.active_stt_plugin.transcribe(f))]
                            if len(active_transcription) > 0:
                                if self.check_for_keyword(active_transcription):
                                    transcription = active_transcription
                            else:
                                visualizations.run_visualization(
                                    "output",
                                    f"<< <noise>"
                                )
                    else:
                        visualizations.run_visualization(
                            "output",
                            f"<  <noise>"
                        )
        except IndexError:
            recordings_available_event.clear()
        if len(transcription) > 0:
            visualizations.run_visualization(
                "output",
                f"<< {transcription}"
            )
            self.handleRequest(transcription)
        return transcription

    def active_listen(self):
        """Active listen does not check for a wakeword.
    `   It should only be used in cases where Naomi has just asked a question"""
        transcription = ""
        recordings_available_event.wait()
        try:
            audio = self.recordings_queue.pop()
            if len(audio)>0:
                with self._write_frames_to_file(audio, None) as f:
                    transcription = [" ".join(self.active_stt_plugin.transcribe(f))]
        except IndexError:
            recordings_available_event.clear()
        if len(transcription) > 0:
            visualizations.run_visualization(
                "output",
                f"<< {transcription}"
            )
        else:
            visualizations.run_visualization(
                "output",
                f"<< <noise>"
            )
        return transcription

    def handle_vad_output(self):
        """This is a thread that converts the audio captured by VAD sequentially
        into a transcript of the words spoken until it runs out of audio to process"""
        while self.Continue:
            try:
                transcription = self.listen()
            except IndexError:
                break

    def say(self, phrase):
        self.actions_queue.appendleft(lambda: self.tts(phrase))
        if not (self.actions_thread and hasattr(self.actions_thread, "is_alive") and self.actions_thread.is_alive()):
            # start the thread
            self.actions_thread = threading.Thread(
                target=self.process_actions
            )
            self.actions_thread.start()

    def process_actions(self):
        while self.Continue:
            try:
                action = self.actions_queue.pop()
                action()
            except IndexError:
                break

    def tts(self, phrase):
        altered_phrase = alteration.clean(phrase)
        if(profile.get_arg('print_transcript')):
            visualizations.run_visualization("output", ">> {}\n".format(phrase))
        with tempfile.SpooledTemporaryFile() as f:
            f.write(self.tts_engine.say(phrase))
            f.seek(0)
            self._output_device.play_fp(f)
            time.sleep(.2)

    def main_loop(self):
        """This is an asynchronous main loop"""
        stt_thread = None
        try:
            while self.Continue:
                # put the audio in a queue and call the stt engine
                self.queue_recording(self.vad_plugin.get_audio())
                if not (stt_thread and hasattr(stt_thread, "is_alive") and stt_thread.is_alive()):
                    # start the thread
                    stt_thread = threading.Thread(
                        target=self.handle_vad_output
                    )
                    stt_thread.start()
        except KeyboardInterrupt:
            self.Continue = False
        visualizations.run_visualization(
            "output",
            "Exiting..."
        )


    def handleRequest(self, utterance):
        handled = False
        intent = self.brain.query(utterance)
        if intent:
            try:
                self._logger.info(intent)
                intent['action'](intent, self)
                handled = True
            except Unexpected as e:
                utterance = e.utterance
            except Exception as e:
                self._logger.error(
                    'Failed to service intent {}: {}'.format(intent, str(e)),
                    exc_info=True
                )
                self.say(self.gettext("I'm sorry."))
                self.say(self.gettext("I had some trouble with that operation."))
                self.say(str(e))
                self.say(self.gettext("Please try again later."))
                handled = True
            else:
                self._logger.debug(
                    " ".join([
                        "Handling of phrase '{}'",
                        "by plugin '{}' completed"
                    ]).format(
                        utterance,
                        intent
                    )
                )
        else:
            self.say_i_do_not_understand()
            handled = True
        return utterance, handled

    def say_i_do_not_understand(self):
        self.say(
            random.choice(
                [  # nosec
                    self.gettext("I'm sorry, could you repeat that?"),
                    self.gettext("My apologies, could you try saying that again?"),
                    self.gettext("Say that again?"),
                    self.gettext("I beg your pardon?"),
                    self.gettext("Pardon?")
                ]
            )
        )

    # If we are using the asynchronous say, so we can hear the "stop"
    # command, what we want to do is put the prompt on the queue, then
    # yield control back to the main conversation loop. When the prompt
    # is spoken in the thread, then we want to return to this point in
    # the main thread and use the blocking listen to listen until we get
    # an actual transcription. That should then be used to determine
    # whether the user responded with one of the expected phrases or not.
    def expect(self, prompt, phrases, name='expect', instructions=None):
        expected_phrases = phrases.copy()
        phrases.extend(
            profile.get_arg("application").brain.get_plugin_phrases(True)
        )
        # set up the special mode so it is pre-generated later
        with self.special_mode(name, phrases):
            pass
        # If "listen_while_talking" is set to true, then we want to create the
        # special mode after saying the prompt. If not, then we want to create
        # the special mode first, so we are ready to listen. Also, there is no
        # need to add anything to the queue if we are not listening while
        # talking.
        if(profile.get_arg('listen_while_talking', False)):
            self.say(prompt)
            self.queue_action(lambda: profile.set_arg('resetmic', True))
            # Now wait for any sounds in the queue to be processed
            while not profile.get_arg('resetmic'):
                transcribed = self.listen()
                handled = False
                if isinstance(transcribed, bool):
                    handled = True
                else:
                    while(" ".join(transcribed) != "" and not handled):
                        transcribed, handled = profile.get_arg('application').conversation.handleRequest(transcribed)
            # Now that we are past the mic reset
            profile.set_arg('resetmic', False)
        else:
            self.say(prompt)
        # Now start listening for a response
        with self.special_mode(name, phrases):
            while True:
                transcribed = self.active_listen()
                if(len(' '.join(transcribed))):
                    # Now that we have a transcription, check if it matches one of the phrases
                    phrase, score = profile.get_arg("application").brain._intentparser.match_phrase(transcribed, expected_phrases)
                    # If it does, then return the phrase
                    self._logger.info("Expecting: {} Got: {}".format(expected_phrases, transcribed))
                    self._logger.info("Score: {}".format(score))
                    if(score > .1):
                        return phrase
                    # Otherwise, raise an exception with the active transcription.
                    # This will break us back into the main conversation loop
                    else:
                        # If the user is not responding to the prompt, then assume that
                        # they are starting a new command. This should mean that the wake
                        # word would be included.
                        if(self.check_for_keyword(transcribed)):
                            raise Unexpected(transcribed)
                        else:
                            # The user just said something unexpected. Remind them of their choices
                            if instructions is None:
                                profile.get_arg("application").conversation.list_choices(expected_phrases)
                            else:
                                self.say(instructions)

    # confirm is a special case of expect which expects "yes" or "no"
    def confirm(self, prompt):
        # default to english
        language = profile.get(['language'], 'en-US')[:2]
        POSITIVE = ['YES', 'SURE', 'YES PLEASE']
        NEGATIVE = ['NO', 'NOPE', 'NO THANK YOU']
        if(language == "fr"):
            POSITIVE = ['OUI']
            NEGATIVE = ['NON']
        elif(language == "de"):
            POSITIVE = ['JA']
            NEGATIVE = ['NEIN']
        phrase = self.expect(
            prompt,
            POSITIVE + NEGATIVE,
            name='confirm',
            instructions=self.gettext(
                "Please respond with Yes or No"
            )
        )
        if phrase in POSITIVE:
            return True
        else:
            return False


    def list_choices(self, choices):
        if len(choices) == 1:
            self.say(self.gettext("Please say {}").format(choices[0]))
        elif len(choices) == 2:
            self.say(self.gettext("Please respond with {} or {}").format(choices[0], choices[1]))
        else:
            self.say(
                self.gettext("Please respond with one of the following: {}").format(choices)
            )
