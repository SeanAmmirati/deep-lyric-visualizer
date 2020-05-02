import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class LyricWeigher(ABC):

    def __init__(self, idx_range=None):
        """An abstract class, used as a schematic for weighing lyrics. This can
            be used to prioritize certain parts of a line, stanza, etc.

        Args:
            idx_range (list, optional): The lyric tokens in a list to consider.
                Defaults to None. For instance, you may only want to consider
                the first and last token.
        """
        self.idx_range = idx_range

    @abstractmethod
    def weigh_subset(self, lyrics, *args, **kwargs):
        """An abstract method -- this method should describe the method for
                weighing the subset of lyric tokens. Should return the
                multiplicative weight for the subset. These should sum to one.

        Args:
            lyrics (list [str]): lyrics to weigh

        """
        pass

    def weigh_lyrics(self, lyrics, *args, **kwargs):
        """Creates an array to store the weights, then weighs the values
        for the subset based on the weigh_subset method.

        Args:
            lyrics (list [str]): A list of tokens of lyrics to weigh

        Returns:
            np.array: An array containing the appropriate weights -- this
                should sum to one.
        """
        ret = np.zeros(len(lyrics))
        ret[self.idx_range] = self.weigh_subset(lyrics, *args, **kwargs)
        return ret


class EqualLyricWeigher(LyricWeigher):
    """Weighs all lyrics equally -- regardless of content.

    Args:
        LyricWeigher (lyrics.LyricWeigher): The abstract LyricWeigher
    """

    def weigh_subset(self, lyrics):
        """Weighs the subset of lyrics equally.

        Args:
            lyrics (list [str]): A list of tokens to weigh.

        Returns:
            np.array: An array containing the weights -- this will be equal
                probability weights, so 1/ the number of lyrics in the subset.
        """

        if not isinstance(lyrics, np.ndarray):
            lyrics = np.array(lyrics)

        subset = lyrics[self.idx_range] if self.idx_range else lyrics

        return np.array([1/len(subset)] * len(subset))


class ConeLyricWeigher(LyricWeigher):

    def __init__(self, idx_range=None, concavity=1):
        """Weighs lyrics in a "cone" structure, with lyrics at the start and
        end of the line being weighed more heavily than those in the center,
        and with symmetric values.

        Args:
            idx_range (list, optional): The lyric tokens in a list to consider.
                Defaults to None. For instance, you may only want to consider
                the first and last token.
            concavity (float, optional): The degree of concavity -- this is the
                exponent for the weights before it is normalized. 0 would weigh
                them all equally. Negative values place more weight on the
                center. Defaults to 1.
        """

        super().__init__(idx_range)
        self.concavity = concavity

    def weigh_subset(self, lyrics):
        """Weighs the subset of lyrics using the cone strategy based on the
        concavity.

        Args:
            lyrics (list [str]): A list of lyric tokens.

        Returns:
            np.array: An array containing the weights for each lyric.
        """
        if not isinstance(lyrics, np.ndarray):
            lyrics = np.array(lyrics)

        subset = lyrics[self.idx_range] if self.idx_range else lyrics
        n_tokens = len(subset)

        weights = np.array([max([(n_tokens - i)/n_tokens, (i + 1)/n_tokens])
                            for i in range(n_tokens)])
        weights = weights ** self.concavity
        weights /= weights.sum()

        return weights


if __name__ == '__main__':
    test_lyrics = ['believe', 'possibility', 'finally', 'happy']

    e_l_w = EqualLyricWeigher()
    c_l_w = ConeLyricWeigher([0, 1, 3], concavity=1)

    print(e_l_w.weigh_lyrics(test_lyrics))
    print(e_l_w.weigh_lyrics(test_lyrics))

    print(c_l_w.weigh_lyrics(test_lyrics))
    print(c_l_w.weigh_lyrics(test_lyrics))
