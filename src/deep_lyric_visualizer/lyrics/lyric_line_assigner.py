import numpy as np
import pandas as pd

from deep_lyric_visualizer.lyrics.lyric_weigher import ConeLyricWeigher, EqualLyricWeigher
from deep_lyric_visualizer.nlp.topic_selector import MaxMaxSelector, MeanMaxSelector
from deep_lyric_visualizer.nlp.wordvector_similarity import (CosineWordVectorSimilarity,
                                                             EuclidWordVectorSimilarity)


class LyricLineAssigner:

    def __init__(self, weighing_type='eq', similarity_metric='cosine',
                 topic_selector_type='max_max', weighing_obj=None,
                 similarity_obj=None, topic_selector_obj=None):
        """A utility that handles the assignment of weights to each lyric in a
        line, stanza, etc for use in a vectorizer.

        Args:
            weighing_type (str, optional): the type of weighing to use, if
                defined by a lyrics.LyricWeigher class.
                One of {'eq', 'cone', 'first', 'last'}.
                For ease of use -- just syntactic sugar for passing the
                appropriate class to weighing_obj argument.
                Defaults to 'eq', which gives all words an equal weight.
            similarity_metric (str, optional): defines the similarity
                between word vectors, as defined by a
                nlp.WordvectorSimilarity object.
                One of {'cosine', 'euclid'}.
                For ease of use -- just syntactic sugar for passing the
                appropriate class to similarity_obj.
                Defaults to 'cosine', which uses the cosine distance between
                the two vectors.
            topic_selector_type (str, optional): defines the topic selector,
                as in how to select a topic given a line of lyrics. One of
                {'mean_max', 'max_max'}. This is defined as a nlp.TopicSelector
                subclass.
                For ease of use -- just syntactic sugar for passing the
                appropriate class to topic_selector.
                Defaults to 'max_max'.
            weighing_obj (lyrics.LyricWeigher, optional): A custom LyricWeigher
                object. Defaults to None.
            similarity_obj (nlp.WordVectorSimilarity, optional): A custom
                WordVectorSimilarity object. Defaults to None.
            topic_selector_obj (nlp.TopicSelector, optional): A custom
                TopicSelector object. Defaults to None.
        """
        if similarity_obj:
            self.similarity = similarity_obj
        elif similarity_metric == 'cosine':
            self.similarity = CosineWordVectorSimilarity()
        elif similarity_metric == 'euclid':
            self.similarity = EuclidWordVectorSimilarity()

        if weighing_obj:
            self.weight = weighing_obj
        elif weighing_type == 'eq':
            self.weight = EqualLyricWeigher()
        elif weighing_type == 'cone':
            self.weight = ConeLyricWeigher()
        elif weighing_type == 'first':
            self.weight = EqualLyricWeigher(0)
        elif weighing_type == 'last':
            self.weight = EqualLyricWeigher(-1)

        if topic_selector_obj:
            self.topic_selector = topic_selector_obj
        elif topic_selector_type == 'max_max':
            self.topic_selector = MaxMaxSelector()
        elif topic_selector_type == 'mean_max':
            self.topic_selector = MeanMaxSelector()

    def assign_line(self, line, candidate_vectors, n=1):
        """Assigns a line of lyrics to a topic.

        Args:
            line (list [np.array]): A list of vectors for a line of lyrics
            candidate_vectors (list [np.array]): A list of candidate vectors
                to consider (the topic vectors)
            n (int, optional): The number of topics to return. Defaults to 1.

        Returns:
            list [int]: A list of indexes representing the appropriate topics
                that were chosen. (tentative)
        """
        weights = self.weight.weigh_lyrics(line)
        similarities = pd.DataFrame([self.similarity.calculate_similarities(
            word, candidate_vectors) for word in line])

        weighted_similarities = weights * similarities.T
        return self.topic_selector.return_selections(weighted_similarities, n)


if __name__ == '__main__':
    lla = LyricLineAssigner()
    test_arr = np.random.normal(2, 1, size=(3, 1000))
    test_vec = [np.array([2, 2, 2]), np.array([3, 4, 3])]
    print(lla.assign_line(test_vec, test_arr, None))

    lla2 = LyricLineAssigner(weighing_type='cone', similarity_metric='euclid')
    print(lla2.assign_line(test_vec, test_arr, 4))
