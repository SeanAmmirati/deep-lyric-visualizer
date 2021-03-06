
import logging
from deep_lyric_visualizer.helpers import setup_logger, dict_assign, find_first_file_with_ext

from deep_lyric_visualizer.generator.generator_object import GeneratorObject

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

from deep_lyric_visualizer.generator.generatorio import PickleGeneratorIO, YAMLGeneratorIO
setup_logger()
logger = logging.getLogger(__name__)


class Tokenizer(GeneratorObject):

    def __init__(self, gen_env=None):

        super().__init__(gen_env)
        self.tokens = None

        self.name = __name__

        self.attrs = ['tokens']

    def tokenize_phrase(self, phrase, process=True):
        tokens = word_tokenize(phrase)

        if process:
            tokens = self._process_tokens(tokens)
            logger.info(f'All tokens processed.')
        else:
            logger.debug(
                'Processing flag is false. Simply returning raw tokens.')

        return tokens

    def _process_tokens(self, tokens):
        logger.debug(f'Tokens: {tokens} -- Processing start...')

        logger.debug(
            f'Tokens: {tokens} -- Lowercasing and removing non alphanumeric tokens.')
        processed_tokens = [w.lower() for w in tokens if w.isalpha()]
        logger.debug(
            f'Tokens: {tokens} -- Lowercased and filtered tokens. Output: {processed_tokens}')

        logger.debug(
            f'Tokens: {tokens} -- Creating stopwords list based on configuration file and common stopwords.')
        stopwords = set(nltk_stopwords.words('english')) | \
            set(self.env.ADDITIONAL_STOPWORDS) - \
            set(self.env.REMOVED_STOPWORDS)
        logger.debug(
            f'Tokens: {tokens} -- Completed stopword generation. Stopwords : {stopwords}')

        logger.debug(f'Tokens: {tokens} -- Removing words in stopwords.')
        selected = [w for w in processed_tokens if w not in stopwords]
        logger.debug(
            f'Tokens: {tokens} -- Removed stopwords tokens. Output: {selected}')

        logger.debug(
            f'Tokens: {tokens} -- Completed processing. Final output: {selected}')
        return selected

    def load(self, songname=None):
        self.genio.load(songname)

    def save(self, songname=None):
        """Exports generated tokens for future use

        Keyword Arguments:
            export_type {'pickle' | 'yaml'}  Type to export to. Pickle may have more overhead in space, but yaml will have higher processing times on load.
            (default: {'pickle'})
        """
        self.genio.save(songname)
