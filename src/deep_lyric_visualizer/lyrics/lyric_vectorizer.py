import os
import yaml
import pickle
import logging
from deep_lyric_visualizer.helpers import setup_logger
from deep_lyric_visualizer.generator.generation_environment import GenerationEnvironment, WikipediaBigGANGenerationEnviornment
import numpy as np

from deep_lyric_visualizer.nlp.vectorizer import Vectorizer
setup_logger()
logger = logging.getLogger(__name__)


class LyricVectorizer(Vectorizer):

    def __init__(self, gen_env=None):
        """A vectorizing utility for lyrics.

        Args:
            gen_env (generator.GenerationEnvironment, optional): A generation
            enviornment object. Defaults to None.
        """
        super().__init__(gen_env)

        self.name = __name__

    def vectorize_token_list(self, token_list):
        """Vectorizes each token in a list of list of of tokens. Memoizes
        this operation to avoid recomputation on identical tokens.

        Args:
            token_list (list [list [str]]): A list of list of tokens, one
            for each line of the song. (For instance
            [["somewhere"]])

        Returns:
            dict: A dictionary with tokens as keys and vectors as values
        """

        logger.info(
            'Vectorizing lists of tokens and adding to word_to_vec dictionary.')
        n_failed = 0
        for tokens in token_list:
            if not tokens:
                logger.warning(f'Empty tokens list. Skipping...')
                continue
            success_pct = self.memoize_vectorize_tokens(tokens)
            if success_pct != 100:
                n_failed += 1
        logger.info(f'All lines\' tokens have been vectorized')
        if n_failed > 0:
            logger.warning('Some words could not be converted.')
        return self.word_to_vec

    def vectorize_line(self, line):
        """Vectorizes a single line of tokens

        Args:
            line (list [str]): A list of tokens

        Returns:
            list [np.array]: A list of arrays (vectors) for each word
        """
        return [self.vectorize_word(w) for w in line]

    def vectorize_lines(self, token_list, start=None, stop=None):
        """Vectorize multiple lines of tokens

        Args:
            token_list (list [list [str]]): A list of list of tokens -- one
                for each line.
            start (int, optional): The line to start at. Defaults to None,
                which will use the first line.
            stop (int, optional): The line to end at. Defaults to None,
                which will use the lat line.

        Returns:
            list [list [np.array]]: A list of list of vectors, one list of
                vectors for each line.
        """
        start = 0 if start is None else start
        stop = len(token_list) if stop is None else stop
        return [self.vectorize_line(l) for l in token_list[start:stop]]

    def vectorize_song(self, token_list):
        """A helper method -- just uses vectorize lines on the whole song.

        Args:
            token_list (list [list [str]]): A list of list of tokens -- one
                for each line.

        Returns:
            list [list [np.array]]: A list of list of vectors, one list of
                vectors for each line.
        """
        return self.vectorize_lines(token_list)

    def load_song(self, songname):
        """Loads the song vectorizer (and results) from a saved location.

        Args:
            songname (str): The name of the song

        Returns:
            NoneType: Returns nothing -- loads the attributes of the saved
                file into memory.
        """
        return self.load(songname)

    def save_song(self, songname):
        """Saves the song vectorizer (and results).

        Args:
            songname (str): The name of the song

        Returns:
            NoneType: Returns nothing -- saves the attributes of the saved
                file into the respective folder.
        """
        return self.save(songname)
