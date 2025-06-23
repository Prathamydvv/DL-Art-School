""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from models.audio.tts.tacotron2.text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

# Devanagari Unicode block: U+0900 to U+097F
# Hindi vowels + consonants (core set)
_letters = 'ऀँंःऄअआइईउऊऋॠऌॡएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळक्षत्रज्ञ'

# No ARPAbet needed for Hindi
_arpabet = []

# Combine all
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
