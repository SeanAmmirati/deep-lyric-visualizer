
import logging
from deep_lyric_visualizer.helpers import setup_logger, _extract_name_from_path

from deep_lyric_visualizer.nlp.tokenizer import Tokenizer
from deep_lyric_visualizer.image_categories.image_category_vectorizer import ImageCategoryVectorizer


setup_logger()
logger = logging.getLogger(__name__)


class ImageCategoryTokenizer(Tokenizer):
    def __init__(self, gen_env=None):
        """The tokenizer used for the image categories. Inherits from the more
        general Tokenizer class in the nlp folder of this package.

        Args:
            Tokenizer (nlp.Tokenizer): The Tokenizer class, from which this
                class inherits.
            gen_env (generator.GenerationEnvironment, optional):
                An instance of the GenerationEnvironment class.
                Defaults to None, which will use th efeautl environment.
        """
        super().__init__(gen_env)

        self.name = __name__ if __name__ != '__main__' else _extract_name_from_path(
            __file__)
        self.image_classes = None
        self.class_tokens = None

        self.attrs = ['class_tokens']

    def tokenize_category(self, category, cat_sep=','):
        """Tokenize a line in the categories of ImageNet. This is useful
            because categories can actually consist of multiple related topics
            in the definition. This will tokenize each sub-category.

        Args:
            category (str): A string, representing the category
            cat_sep (str, optional): The seperator which splits the topics.
            Defaults to ','.

        Returns:
            list: A list of lists, with tokens for each topic
        """
        seperated = category.split(cat_sep)
        return [self.tokenize_phrase(phrase) for phrase in seperated]

    def load_image_classes(self):
        """Loads the image classes using the environment instance.
        """
        self.image_classes = self.env.read_id_to_img_class()

    def tokenize_image_classes(self):
        """Tokenizes the image classes and assigns them to the class_tokens
        attribute.
        """

        if not self.image_classes:
            logger.debug('No image classes loaded. Attempting to load now.')
            self.load_image_classes()

        self.class_tokens = {id_: self.tokenize_category(
            cat) for id_, cat in self.image_classes.items()}


if __name__ == '__main__':
    im_token = ImageCategoryTokenizer()
    # im_token.tokenize_image_classes()
    # im_token.save()
    im_token.load()

    im_vectorizer = ImageCategoryVectorizer()
    print(im_vectorizer.vectorize_categories(im_token.class_tokens))
    im_vectorizer.save()
