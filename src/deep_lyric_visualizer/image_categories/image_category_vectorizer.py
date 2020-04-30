
import logging
from helpers import setup_logger, _extract_name_from_path
from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)
from deep_lyric_visualizer.generator.generatorio import PickleGeneratorIO, YAMLGeneratorIO

from nlp.vectorizer import Vectorizer
import numpy as np
setup_logger()
logger = logging.getLogger(__name__)


class ImageCategoryVectorizer(Vectorizer):
    def __init__(self, gen_env=None):
        """A vectorizer specific to image categories. Inherits from the
        Vectorizer class in the nlp section of this package.

        Args:
            Vectorizer (nlp.Vectorizer): the Vectorizer class, from which this
                class inherits.
            gen_env (generator.GenerationEnvironment, optional): a
                GenerationEnvironment instance. If None, uses the default .
                Defaults to None.
        """
        super().__init__(gen_env)

        self.name = __name__ if __name__ != '__main__' else _extract_name_from_path(
            __file__)
        self.vectorized_dict = None

        self.attrs = ['vectorized_dict']

    def _mean_strategy(self, category_tokens):
        """Defines the so-called 'mean' strategy for vectorizing a list of
        list of tokens for a category. Each sub-category or topic is treated
        first, averaging the embeddings for the tokens in that category.
        Then, the results from each sub-category/topic are averaged together.

        Args:
            category_tokens (list): This is a list of lists, one list of tokens
                for each topic in the category.

        Returns:
            np.array: A so-called category vector, an embedding for the
                category.
        """

        wordvec_sum = np.zeros(self.env.wordvec_dim)
        n_phrases = 0

        for tokens in category_tokens:
            n = len(tokens)
            if n == 0:
                continue

            vec = np.zeros(self.env.wordvec_dim)
            n_vectorizable_phrases = 0
            for token in tokens:
                try:
                    vectorized = self.vectorize_word(token)
                except KeyError:
                    pass
                else:
                    n_vectorizable_phrases += 1
                    vec += vectorized
            if n_vectorizable_phrases == 0:
                continue
            else:
                n_phrases += 1
            vec = vec / n_vectorizable_phrases
            wordvec_sum += vec
        mean_wordvec = (
            wordvec_sum / n_phrases) if n_phrases != 0 else wordvec_sum

        return mean_wordvec

    def vectorize_category(self, category_tokens, strategy='mean'):
        """Handles the vectorization of a cateogry by a particular strategy.
        At the moment, the only considered strategy is the mean strategy.

        Args:
            category_tokens (list [list [str]]): This is a list of lists,
                one list of tokens for each topic in the category.
            strategy (str, optional): One of {"mean"}. The strategy to use
                Currently only the mean strategy is supported.
                Defaults to 'mean'.

        Returns:
            np.array: An array with the vector representing the category
        """
        if strategy == 'mean':
            return self._mean_strategy(category_tokens)

    def vectorize_categories(self, categories_tokens, strategy='mean'):
        """Vectorize a set of categories given their lists of lists of tokens.

        Args:
            categories_tokens (dict): A dictionary representing the id number
            for a category to the list of lists of tokens for that category.
            strategy (str, optional): One of {"mean"}. The strategy to use
                Currently only the mean strategy is supported.
                Defaults to 'mean'.

        Returns:
            dict: Dictionary with embeddings for each category_id.
        """
        self.vectorized_dict = {id_: self.vectorize_category(
            category) for id_, category in categories_tokens.items()}
        return self.vectorized_dict


if __name__ == '__main__':
    im_vec = ImageCategoryVectorizer()
    im_vec.load()
    print(im_vec.vectorized_dict)
