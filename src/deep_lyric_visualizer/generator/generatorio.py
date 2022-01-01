from abc import ABC, abstractmethod
from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)
import logging
from deep_lyric_visualizer.helpers import setup_logger

import os

import pickle
import yaml
import numpy as np
setup_logger()
logger = logging.getLogger(__name__)


class GeneratorIO(ABC):

    def __init__(self, obj):
        """An class for handling In-Out operations of generator objects.

        Args:
            obj (GeneratorObject): a Generator object to save data from or load
            data into

        Raises:
            ValueError: Raises value error when passed an object which is not
            an environment object.
        """

        self.env = obj.env
        self.word_embedder = self.env.word_embedder()

        self.word_to_vec = {}
        self.obj = obj

    @property
    def attrs(self):
        """A property representing the attributes of the object. As this
        changes over time, it is a property as an ease of reference. However,
        this has no custom getter, setter or delete functions.

        Returns:
            list: A list of attributes of the object
        """
        return self.obj.attrs

    def get_saving_location(self, songname=None):
        """Based on the objects name, return the appropriate save location for
        a songs' data.

        Args:
            songname ([type], optional): ame of the song, if necessary.
            Defaults to None.

        Returns:
            str: The path of the save location of the file for that object.
        """

        if self.obj.name == 'lyric_tokenizer':
            self.save_loc = self.env.song_lyric_filename(songname)

        if self.obj.name == 'lyric_vectorizer':
            self.save_loc = self.env.song_embeddings_filename(songname)

        if self.obj.name == 'image_category_vectorizer':
            self.save_loc = self.env.class_embeddings_filename()

        if self.obj.name == 'image_category_tokenizer':
            self.save_loc = self.env.class_token_filename()

        if self.obj.name == 'lyrics':
            self.save_loc = self.env.complete_lyrics_filename(songname)

        return self.save_loc

    def make_save_locations(self, songname=None):
        """Create the save locations in the directory structure, if they do
        not exist.

        Args:
            songname (string, optional): name of the song, if necessary.
            Defaults to None.
        """

        full_path = self.get_saving_location(
            songname)
        dir_name = os.path.dirname(full_path)
        dir_hierarchy = []
        while dir_name:
            dir_hierarchy.append(dir_name)
            dir_name = os.path.dirname(dir_name)

        reversed_hierarchy = reversed(dir_hierarchy)

        for p in reversed_hierarchy:
            if not os.path.exists(p):
                logging.debug(f'No directory at {p}. Creating new directory.')
                os.mkdir(p)

    def transform_save(self, attr):
        """Transforms an attribute if necessary for saving. This is an
        option for subclasses, but for the general case simply returns the
        attribute.

        Args:
            attr (Object): an attribute that needs to be transformed

        Returns:
            Object: the expected output for saving.
        """
        return attr

    def transform_load(self, attr):
        """Transforms an attribute if necessary for loading. This is an
        option for subclasses, but for the general case simply returns the
        attribute. Ideally would be the inverse of transform_save.

        Args:
            attr (Object): an attribute that needs to be transformed

        Returns:
            Object: the expected output for loading.
        """
        return attr

    def save(self, songname=None):
        """Save the object to the appropriate location.

        Args:
            songname (str, optional): The name of the song. Defaults to None.
        """

        attrs = [getattr(self.obj, attr, None) for attr in self.attrs]

        transformed_attrs = [self.transform_save(x) for x in attrs]
        self.make_save_locations(songname)
        logger.debug(f'Saving {transformed_attrs} to {self.save_loc}')
        self.save_to_file(transformed_attrs, self.save_loc)
        logger.debug(
            f'Saved information for {self.obj.name} to {self.save_loc}')

    def load(self, songname=None):
        """Load the object to the appropriate location.

        Args:
            songname (str, optional): The name of the song. Defaults to None.
        """
        self.save_loc = self.get_saving_location(songname)
        try:
            res = self.load_from_file(self.save_loc)
        except FileNotFoundError:
            logger.error(f'No file to load in {self.save_loc}')
            raise
        for i, r in enumerate(res):
            logger.debug(f'Loading {r} into {self.attrs[i]} attribute.')
            setattr(self.obj, self.transform_load(self.attrs[i]), r)

        logger.info(
            f'Loaded information for {self.obj.name} from {self.save_loc}')

    @abstractmethod
    def load_from_file(self, where):
        """An abstract method -- this describes the actual loading process.
        This will vary based on the file type (for instance, pickle, yaml, etc)

        Args:
            where (str): The path to load the attributes from
        """
        pass

    @abstractmethod
    def save_to_file(self, x, where):
        """An abstract method -- this describes the actual saving process.
        This will bary based on the file type (for instance, pickle, yaml, etc)

        Args:
            x (Object): An object to save
            where (str): The path to save the attributes to
        """
        pass


class PickleGeneratorIO(GeneratorIO):

    def save_to_file(self, x, where):
        """The specific saving procedure for pickle files.

        Args:
            x (Object): An object to save
            where (str): The path to save the attributes to
        """
        with open(where, 'wb') as f:
            pickle.dump(x, f)

    def load_from_file(self, where):
        """The specific loading procedure for pickle files.

        Args:
            where (str): The path to load the attributes from

        Returns:
            Object: The object contained in the pickle file
        """
        with open(where, 'rb') as f:
            return pickle.load(f)


class YAMLGeneratorIO(GeneratorIO):

    def save_to_file(self, x, where):
        """The specific saving procedure for yaml files.

        Args:
            x (Object): An object to save
            where (str): The path to save the attributes to
        """
        with open(where, 'w') as f:
            yaml.dump(x, f)

    def load_from_file(self, where):
        """The specific loading procedure for pickle files.

        Args:
            where (str): The path to load the attributes from

        Returns:
            Object: The object contained in the pickle file
        """
        with open(where, 'r') as f:
            return yaml.load(f)

    def transform_save(self, attr):
        """When saving an array, we must store this as a list in the yaml file.

        Args:
            attr (Object): Any object, but only performed when it is an array.

        Returns:
            Object: The original type of the passed attribute, unless it was
            an array, in which case it will be returned as a list.
        """
        if isinstance(attr, np.array):
            return attr.tolist()
        else:
            return attr

    def transform_load(self, attr):
        """When loading a list, we will transform this into an array.

        Args:
            attr (Object): Any object, but only performed when it is an array.

        Returns:
            Object: The original type of the passed attribute, unless it was
            a list, in which case it will be returned as an array.
        """
        if isinstance(attr, list):
            return np.array(attr)
        else:
            return attr
