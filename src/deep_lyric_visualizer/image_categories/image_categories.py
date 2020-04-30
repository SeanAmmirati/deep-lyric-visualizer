import logging
from deep_lyric_visualizer.helpers import setup_logger

from image_categories.image_category_tokenizer import ImageCategoryTokenizer
from image_categories.image_category_vectorizer import ImageCategoryVectorizer

from deep_lyric_visualizer.generator.generator_object import GeneratorObject

setup_logger()
logger = logging.getLogger(__name__)


class ImageCategories(GeneratorObject):

    def __init__(self, tokenizer=None, vectorizer=None, gen_env=None):
        """This class handles the tokenizing and vectorizing of the image
        categories. These categories are the categories used by imagenet to
        classify pictures. These are then used to compare to the lyrics
        to select for the GAN.

        Args:
            tokenizer (Tokenizer, optional): The Tokenizer to use, from the
                nlp module inside of this package. Defaults to None.

            vectorizer (Vectorizer, optional): The Vectorizer to use, from the
                nlp module inside of this package. Defaults to None.

            gen_env (GenerationEnvironment, optional): The Generation
                Environment to use, from the generator module inside of this
                package. Defaults to None.

        """
        super().__init__(gen_env)
        self.img_cat_tokenizer = ImageCategoryTokenizer(
            self.env) if not tokenizer else tokenizer
        self.img_cat_vectorizer = ImageCategoryVectorizer(
            self.env) if not vectorizer else vectorizer

        self._tokens = None
        self._vectors = None
        self._strings = None

    @property
    def strings(self):
        """A property representing the string representations of the
        image classes.

        Returns:
            dict: A dictionary with the ImageNet class_ids to class names.
        """
        if not self._strings:
            self.img_cat_tokenizer.load_image_classes()
            self._strings = self.img_cat_tokenizer.image_classes
        return self._strings

    @property
    def tokens(self, save=True):
        """A property representing the tokens extracted from the category names.
        Will try to load this file if it exists -- otherwise, uses the
        Tokenizer to build the tokens manually.

        Args:
            save (bool, optional): If manually generating the tokens, whether
                you wish to save them in the default location. Defaults to
                True.

        Returns:
            dict: A dictionary containing the ImageNet image class id to the
                tokens associated with that image class id. The values will be
                lists of lists of tokens.
        """
        if not self._tokens:
            try:
                self.img_cat_tokenizer.load()
            except FileNotFoundError:
                self.img_cat_tokenizer.tokenize_image_classes()
                if save:
                    self.img_cat_tokenizer.save()
            self._tokens = self.img_cat_tokenizer.class_tokens
        return self._tokens

    @property
    def vectors(self, save=True):
        """A property representing the category vectors extracted from the
        category names. Will try to load this file if it exists -- otherwise,
        uses the Vectorizer to build the vectors manually.

        A single vector is generated for each category.

        Args:
            save (bool, optional): If manually converting to vectors, whether
                you wish to save them in the default location. Defaults to
                True.



        Returns:
            dict: A dictionary containing the ImageNet image class id to the
                word vectors associated with that image class id. The values
                will be a single category vector meant to represent that
                category.
        """
        if not self._vectors:
            try:
                self.img_cat_vectorizer.load()
            except FileNotFoundError:
                self.img_cat_vectorizer.vectorize_categories(self.tokens)
                if save:
                    self.img_cat_vectorizer.save()
            self._vectors = self.img_cat_vectorizer.vectorized_dict

        return self._vectors

    def find_category_string_by_id(self, id_):
        """A utility method to return the name of a category by id.

        Args:
            id_ (int): The ImageNet ID number

        Returns:
            str: the category name
        """
        return self.strings[id_]

    def find_category_tokens_by_id(self, id_):
        """A utility method to return the tokens in a category by id.

        Args:
            id_ (int): The ImageNet ID number

        Returns:
            list: the tokens present in the category
        """
        return self.tokens[id_]

    def find_category_vector_by_id(self, id_):
        """A utility method to return the category vectors in a category by id.

        Args:
            id_ (int): The ImageNet ID number

        Returns:
            list: the vector to represent the category
        """
        return self.vectors[id_]


if __name__ == '__main__':
    img_cats = ImageCategories()

    string = img_cats.find_category_string_by_id(3)
    tokens = img_cats.find_category_tokens_by_id(3)
    vec = img_cats.find_category_vector_by_id(3)

    print(string, tokens, vec)
