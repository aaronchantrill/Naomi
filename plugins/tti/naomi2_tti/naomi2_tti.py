# -*- coding: utf-8 -*-
import logging
import os
import re
from jiwer import wer
from naomi import paths
from naomi import plugin
from naomi import profile
import pdb
from pprint import pprint
import random

# The original intent parser for Jasper was very simple. The intents would
# line up and when an utterance would come in, Naomi would go down the line
# of plugins and ask each in turn if they could service the utterance.
# Convert a word ("word") to a keyword ("{word}")
def to_keyword(word):
    return "{}{}{}".format("{", word, "}")


class Naomi2TTIPlugin(plugin.TTIPlugin):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger(__name__)

    # self
    #   .intent_map
    #       ['intents'][intent]['templates'][template]
    #   .keywords
    #       [intent]
    #   .words
    #       [word]
    def add_intents(self, intents):
        for intent in intents:
            # this prevents collisions between intents by different authors
            intent_base = intent
            intent_inc = 0
            locale = profile.get("language")
            while intent in self.intent_map['intents']:
                intent_inc += 1
                intent = "{}{}".format(intent_base, intent_inc)
            if('locale' in intents[intent_base]):
                # If the selected locale is not available, try matching just
                # the language ("en-US" -> "en")
                if(locale not in intents[intent_base]['locale']):
                    for language in intents[intent_base]['locale']:
                        if(language[:2] == locale[:2]):
                            locale = language
                            break
            if(locale not in intents[intent_base]['locale']):
                raise KeyError("Language not supported")
            if('keywords' in intents[intent_base]['locale'][locale]):
                if intent not in self.keywords:
                    self.keywords[intent] = {}
                for keyword in intents[intent_base]['locale'][locale]['keywords']:
                    if keyword not in self.keywords[intent]:
                        self.keywords[intent][keyword] = []
                    self.keywords[intent][keyword].extend([word.upper() for word in intents[intent_base]['locale'][locale]['keywords'][keyword]])
            self.intent_map['intents'][intent] = {
                'action': intents[intent_base]['action'],
                'name': intent_base,
                'templates': [],
                'words': {}
            }
            for phrase in intents[intent_base]['locale'][locale]['templates']:
                # Save the phrase so we can search for undefined keywords
                # Convert the template to upper case and expand contractions
                clean_phrase = self.cleantext(phrase)
                # At this point, I want to make a list of variations for each
                # possible contraction.
                contraction_phrases = self.getcontractions(clean_phrase)
                #pdb.set_trace()
                self.intent_map['intents'][intent]['templates'].append(clean_phrase)
                for word in phrase.split():
                    if not self.is_keyword(word):
                        word = word.upper()
                    try:
                        self.intent_map['intents'][intent]['words'][word] += 1
                    except KeyError:
                        self.intent_map['intents'][intent]['words'][word] = 1
                        self._logger.info(f"Adding '{word}' to '{intent}'")
                    # keep a list of the intents a word appears in
                    try:
                        self.words[word].update({intent: True})
                    except KeyError:
                        self.words[word] = {intent: True}

    def train(self):
        # Here we want to go through a list of all the words in all the intents
        # and get a count of the number of intents the word appears in, then
        # divide the weight of every instance by the total number of times it
        # appears in the templates. That way a word that appears a lot (like
        # "what") will get a much lower weight
        wordcounts = {}
        for intent in self.intent_map['intents']:
            for word in self.intent_map['intents'][intent]['words']:
                if word in wordcounts:
                    wordcounts[word] += 1
                else:
                    wordcounts[word] = 1
        for word in wordcounts:
            # set a count for each word
            self.words[word] = wordcounts[word]
        self.trained = True

    def get_plugin_phrases(self, passive_listen=False):
        phrases = []
        # include the keyword, otherwise
        if(passive_listen):
            keywords = profile.get(["keyword"])
            if not (isinstance(keywords, list)):
                keywords = [keywords]
            phrases.extend([word.upper() for word in keywords])
        # Include any custom phrases (things you say to Naomi
        # that don't match plugin phrases). Otherwise, there is
        # a high probability that something you say will be
        # interpreted as a command. For instance, the
        # "check_email" plugin has only "EMAIL" and "INBOX" as
        # standard phrases, so every time I would say
        # "Naomi, check email" Naomi would hear "NAOMI SHUT EMAIL"
        # and shut down.
        custom_standard_phrases_file = paths.data(
            "standard_phrases",
            "{}.txt".format(profile.get(['language'], 'en-US'))
        )
        if(os.path.isfile(custom_standard_phrases_file)):
            with open(custom_standard_phrases_file, mode='r') as f:
                for line in f:
                    phrase = line.strip()
                    if phrase:
                        phrases.append(phrase.upper())

        for intent in self.intent_map['intents']:
            if('templates' in self.intent_map['intents'][intent]):
                templates = self.intent_map['intents'][intent]['templates']
                if(intent in self.keywords):
                    keywords = self.keywords[intent]
                    for keyword in keywords:
                        # This will not replace keywords that do not have a list associated with them, like regex and open keywords
                        # print("Replacing {} with words from {} in templates".format(keyword,keywords[keyword]))
                        for template in templates:
                            if(to_keyword(keyword) in template):
                                templates.extend([template.replace(to_keyword(keyword), word.upper()) for word in keywords[keyword]])
                            # Now that we have expanded every instance of keyword in templates, delete any template that still contains keyword
                            templates = [template for template in templates if not to_keyword(keyword) in template]
                phrases.extend(templates)
        return sorted(phrases)

    def determine_intent(self, phrase):
        phrase = self.cleantext(phrase)
        score = {}
        allvariants = {phrase: {}}
        # replace any keyword found in the utterance with the name of the keyword group.
        # This way if the user says "I am happy" and we have the following
        # intents:
        #    'HowAreYouIntent': {
        #        'locale': {
        #            'en-US': {
        #                'keywords': {
        #                    'MoodKeyword': [
        #                        'HAPPY',
        #                        'SAD',
        #                        'ANGRY',
        #                        'SCARED',
        #                        'EXCITED'
        #                    ]
        #                },
        #                'templates': [
        #                    "I AM {MoodKeyword}",
        #                ]
        #            }
        #        },
        #        'action': self.handle
        #    }
        #    'SetNaomiMoodIntent': {
        #        'locale': {
        #            'en-US': {
        #                'keywords': {
        #                    'MoodKeyword': [
        #                        'HAPPY',
        #                        'SAD',
        #                        'ANGRY',
        #                        'SCARED',
        #                        'EXCITED'
        #                    ]
        #                },
        #                'templates': [
        #                    "YOU ARE {MoodKeyword}",
        #                ]
        #            }
        #        },
        #        'action': self.handle
        #    }
        # we will end up with variants:
        #   "I AM HAPPY": {}
        #   "I AM {MoodKeyword}": {MoodKeyword: ['HAPPY']}
        # In most cases, the presence of a keyword would be a strong indicator
        # that the associated intent is intended, although not always. For
        # instance, if there is also an intent like this:
        #    'MPDIntent': {
        #        'locale': {
        #            'en-US': {
        #                'keywords': {
        #                    'Playlist': [
        #                        "HAPPY",
        #                        "I'M SO EXCITED"
        #                    ]
        #                },
        #                'templates': [
        #                    "Play {PlayList}",
        #                ]
        #            }
        #        },
        #        'action': self.handle
        #    }
        # we will end up with variants:
        #   "I AM HAPPY": {}
        #   "I AM {MoodKeyword}": {MoodKeyword: ['HAPPY']}
        #   "I AM {Playlist}": {Playlist: ['HAPPY']}
        # In both of these cases, the "I AM" part of the request, despite
        # the fact that both words are very common, will determine the
        # intent.
        for intent in self.keywords:
            variants = {phrase: {}}
            for keyword in self.keywords[intent]:
                for word in self.keywords[intent][keyword]:
                    count = 0  # count is the index of the match we are looking for
                    countadded = 0  # keep track of variants added for this count
                    while True:
                        added = 0  # if we get through all the variants without
                        # adding any new variants, then increase the count.
                        for variant in variants:
                            # subs is a list of substitutions
                            # We create a copy so we can only add a new
                            # substitution to the list if it does not already
                            # exist
                            subs = dict(variants[variant])
                            # check and see if we can make a substitution and
                            # generate a new variant.
                            new = self.replacenth(word, "{}{}{}".format('{', keyword, '}'), variant, count)
                            if new not in variants:
                                try:
                                    subs[keyword].append(word)
                                except KeyError:
                                    subs[keyword] = [word]
                                # print(subs[keyword])
                                # print()
                                variants[new] = subs
                                # pprint(variants)
                                added += 1
                                countadded += 1
                                # start looping over variants again
                                break
                        # check if we were able to loop over all the variants
                        # without creating any new ones
                        if added == 0:
                            if countadded == 0:
                                break
                            else:
                                count += 1
                                countadded = 0
            allvariants.update(variants)
        # pdb.set_trace()
        # Now calculate a total score for each variant
        variantscores = {}
        for variant in allvariants:
            self._logger.debug("************VARIANT**************")
            self._logger.debug(variant)
            variantscores[variant] = {}
            words = variant.split()
            intentscores = {}
            # pdb.set_trace()
            for intent in self.intent_map['intents']:
                self._logger.debug(f"Intent: {intent}")
                intentscores[intent] = 0
                # pprint(self.intent_map['intents'][intent]['words'])
                # build up a score based on the words that match.
                for template in self.intent_map['intents'][intent]['templates']:
                    self._logger.debug(f"Scoring template: {template}")
                    score = 0
                    template_words = template.split()
                    # Split the template into words, and get the total words
                    # that match between the template and the variant.
                    for word in words:
                        self._logger.debug(f"Scoring word: {word}")
                        if word in template_words:
                            score += 1/self.words[word] # Add 1/count, more popular words have less weight
                            self._logger.debug(f"Score: {score}")
                    if score > intentscores[intent]:
                        intentscores[intent]=score
                self._logger.debug(f"{intent}: {score}")
            # Take the intent with the highest score
            bestintent = max(intentscores, key=intentscores.get)
            bestscore = intentscores[bestintent]
            # Check if there are multiple intents with the same score.
            intents = [k for k,v in intentscores.items() if v == bestscore]
            if len(intents) > 1:
                # Choose one at random
                self._logger.info(f"Choosing at random from {intents}") 
                bestintent = random.choice(intents)
            variantscores[variant] = {
                'intent': bestintent,
                'input': phrase,
                'score': bestscore,
                'matches': allvariants[variant],
                'action': self.intent_map['intents'][bestintent]['action']
            }
        bestvariant = max(variantscores, key=lambda key: variantscores[key]['score'])
        # find the template with the smallest levenshtein distance
        templates = {}
        for template in self.intent_map['intents'][bestintent]['templates']:
            templates[template] = wer(template, variant)
        besttemplate = min(templates, key=templates.get)
        # The next thing we have to do is match up all the substitutions
        # that have been made between the template and the current variant
        # This is so that if there are multiple match indicators we can eliminate
        # the ones that have matched.
        # Consider the following:
        #   Team: ['bengals','patriots']
        #   Template: will the {Team} play the {Team} {Day}
        #   Input: will done browns play the bengals today
        #   Input with matches: will done browns play the {Team} {Day}
        #   Matches: {Team: bengals, Day: today}
        # Obviously there is a very low Levenshtein distance between the template
        # and the input with matches, but it's not that easy to figure out which
        # Team in the template has been matched. So loop through the matches and
        # words and match the word with each possible location in the template
        # and take the best as the new template.
        #   Template1: will the bengals play the {Team} {Day}
        #   input: will done browns play the bengals {Day}
        #   distance: .42
        #
        #   Template2: will the {Team} play the bengals {Day}
        #   input: will done browns play the bengals {Day}
        #   distance: .28
        #
        # since we are looking for the smallest distance, Template2 is obviously
        # a better choice.
        # print("Best variant: {}".format(bestvariant))
        # print("Best template: {}".format(besttemplate))
        currentvariant = bestvariant
        currenttemplate = besttemplate
        for matchlist in variantscores[bestvariant]['matches']:
            for word in variantscores[bestvariant]['matches'][matchlist]:
                # Substitute word into the variant (we know this matches the
                # first occurrance of {matchlist})
                currentvariant = bestvariant.replace(
                    "{}{}{}".format('{', matchlist, '}'),
                    word,
                    1
                )
                templates = {}
                # Get a count of the number of matches for the
                # current matchlist in template
                possiblesubstitutions = currenttemplate.count(
                    '{}{}{}'.format('{', matchlist, '}')
                )
                # We don't actually know if there are actually any
                # substitutions in the template
                if(possiblesubstitutions > 0):
                    for i in range(possiblesubstitutions):
                        currenttemplate = self.replacenth(
                            '{}{}{}'.format('{', matchlist, '}'),
                            word,
                            currenttemplate,
                            i + 1
                        )
                        templates[currenttemplate] = wer(
                            currentvariant,
                            currenttemplate
                        )
                    currenttemplate = min(
                        templates,
                        key=lambda key: templates[key]
                    )
        # Now that we have a matching template, run through a list of all
        # substitutions in the template and see if there are any we have not
        # identified yet.
        substitutions = re.findall(r'{(.*?)}', currenttemplate)
        if(substitutions):
            for substitution in substitutions:
                subvar = "{}{}{}".format('{', substitution, '}')
                # So now we know that we are missing the variable contained
                # in substitution.
                # What we have to do now is figure out where in the string
                # to insert that variable in order to minimize the levenshtein
                # distance between bestvariant and besttemplate
                variant = currentvariant.split()
                variant.append("<END>")
                template = currenttemplate.split()
                template.append("<END>")
                n = len(variant) + 1
                m = len(template) + 1
                # Find out which column contains the first instance of
                # substitution
                s = template.index(subvar) + 1
                match = []
                a = []
                for i in range(n + 1):
                    a.append([1] * (m + 1))
                    a[i][0] = i
                for j in range(m + 1):
                    a[0][j] = j
                for i in range(1, n):
                    for j in range(1, m):
                        if(variant[i - 1] == template[j - 1]):
                            c = 0
                        else:
                            c = 1
                        a[i][j] = c
                # examine the resulting list of matched words
                # to locate the position of the unmatched keyword
                matched = ""
                for i in range(1, n - 1):
                    if(a[i - 1][s - 1] == 0):
                        # the previous item was a match
                        # so start here and work to the right until there is
                        # another match
                        k = i
                        start = k
                        end = n - 1
                        compare = [k]
                        compare.extend([1] * (m))
                        while((a[k] == compare) and (k < (n))):
                            match.append(variant[k - 1])
                            k += 1
                            compare = [k]
                            compare.extend([1] * (m))
                            end = k
                        matched = " ".join(match)
                        substitutedvariant = variant[:start]
                        substitutedvariant.append(subvar)
                        substitutedvariant.extend(variant[end:])
                        break
                    elif(a[i + 1][s + 1] == 0):
                        # the next item is a match, so start working backward
                        k = i
                        end = k
                        start = 0
                        compare = [k]
                        compare.extend([1] * (m))
                        while(a[k] == compare):
                            match.append(variant[k - 1])
                            k -= 1
                            compare = [k]
                            compare.extend([1] * (m))
                            start = k
                        matched = " ".join(reversed(match))
                        substitutedvariant = variant[:start]
                        substitutedvariant.append(subvar)
                        substitutedvariant.extend(variant[end:])
                        break
                if(len(matched)):
                    try:
                        variantscores[bestvariant]['matches'][substitution].append(matched)
                    except KeyError:
                        variantscores[bestvariant]['matches'][substitution] = [matched]
        return {
            self.intent_map['intents'][variantscores[bestvariant]['intent']]['name']: {
                'action': variantscores[bestvariant]['action'],
                'input': phrase,
                'matches': variantscores[bestvariant]['matches'],
                'score': variantscores[bestvariant]['score']
            }
        }
