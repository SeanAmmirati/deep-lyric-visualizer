
import logging
from deep_lyric_visualizer.helpers import setup_logger, dict_assign, find_first_file_with_ext

from deep_lyric_visualizer.generator.generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
from nlp.tokenizer import Tokenizer

import re
import os

import pylrc


from deep_lyric_visualizer.generator.generatorio import PickleGeneratorIO, YAMLGeneratorIO
setup_logger()
logger = logging.getLogger(__name__)


class LyricTokenizer(Tokenizer):

    def __init__(self, gen_env=None):
        """A utility for tokenizing lyrics of a song

        Args:
            gen_env (generator.GenerationEnvironment, optional):
                A GenerationEnvironment object. Defaults to None.
        """
        super().__init__(gen_env)

        self.lrc_str = None
        self.lrc_obj = None
        self.lyric_list = None
        self.tokens_list = None
        self.name = __name__

        self.attrs = ['tokens_list', 'lrc_str', 'lyric_list', 'lrc_obj']

    def tokenize_lyrics(self, songname, process=True):
        """Tokenizes the lyrics, given a song name, and returns the tokens
        in a list.

        Args:
            songname (str): The name of the song to tokenize
            process (bool, optional): Whether to process the tokens
                or simply use them as is (as space-seprated values in a
                sentence). Usually, you want process to be True.
                Defaults to True.

        Returns:
            list list[str]: A list of lists of tokens, one for each line.
        """

        self.lrc_str = self.env.read_lrc_file(songname)
        self.lrc_obj = pylrc.parse(self.lrc_str)

        self.lyric_list = self.lrc_to_lyric_list(self.lrc_obj)
        logger.info(f'Tokenizing lyrics for song {songname}')

        tokens_list = [self.tokenize_phrase(lyric_line, process=process)
                       for lyric_line in self.lyric_list]
        logger.debug('Generated token list from lines of lrc file.')

        self.tokens_list = tokens_list
        return tokens_list

    def lrc_to_lyric_list(self, lrc_obj):
        """Converts a lrc object to a lyric list.

        Args:
            lrc_obj (pylrc.lrc): A lrc object to extract lyrics from

        Returns:
            list: A list of lyrics in the same format as from a text file.
        """
        logger.debug('Converting lrc file to list of lyric strings.')

        return [x.text for x in lrc_obj]
