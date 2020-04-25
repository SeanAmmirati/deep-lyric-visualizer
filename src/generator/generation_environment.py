import logging
import os
from abc import ABC, abstractmethod

import yaml
from pytorch_pretrained_biggan import BigGAN
from wikipedia2vec import Wikipedia2Vec

from helpers import dict_assign, setup_logger, find_first_file_with_ext

setup_logger()
logger = logging.getLogger(__name__)


class GenerationEnvironment(ABC):

    def __init__(self, cfg=None):
        """A class handling the filesystem and loading of external files for
        the generation project.

        Args:
            cfg (str, optional): A path to a custom configuration file. This
            file should be in a similar structure to the one found in
            config/default_cfg.yaml. Check here for context.
            Defaults to None, which will use the deafault configuration file.

        Raises:
            ValueError: Raised when a file is not found at the location
            specified for the configuration.
        """
        logger.debug('Started initialization of GenerationEnvironment')

        if cfg:
            logger.info(f'Using {cfg} as configuration for tokenizer.')
            self._config_file_loc = cfg
        else:
            logger.info(
                'No configuration supplied. Using default configuration file.')
            self._config_file_loc = 'config/default_cfg.yaml'

        logger.info(f'Loading configuration from {self._config_file_loc}')
        if os.path.exists(self._config_file_loc):
            with open(self._config_file_loc, 'r') as f:
                try:
                    self.cfg = yaml.load(f)
                except Exception as e:
                    logger.error(
                        'Could not load configuration file. Ensure that the '
                        f'file at {self._config_file_loc} is in the correct '
                        'format')
                    raise
                else:
                    logger.debug(
                        'Succesfully loaded yaml file at '
                        f'{self._config_file_loc}.')
        else:
            logger.error(
                f'File does not exist at {self._config_file_loc}. '
                'Ensure that it has been entered correctly')
            raise ValueError

        logger.debug(
            'Assigning key value pairs from configuration dictionary to '
            'attributes of the instance')
        dict_assign(self, self.cfg)
        self.wordvec_dim = 0

    def song_lyric_dir(self, songname):
        """Returns the full path of a particular song's lyric directory.

        Args:
            songname (str): The songname to find the directory for.

        Returns:
            str: The directory containing lyric data for the song.
        """
        return os.path.join(self.LYRIC_PATH, songname)

    def song_embeddings_dir(self, songname):
        """Returns the full path of a particular song's embedding directory.

        Args:
            songname (str): The songname to find the directory for.

        Returns:
            str: The directory containing embedding data for the song.
        """
        return os.path.join(self.SONG_EMBEDDING_PATH, songname)

    def song_lyric_filename(self, songname):
        """Returns the full path of a particular song's token file.

        Args:
            songname (str): The songname to find the directory for.

        Returns:
            str: The path of the file containing lyric data for the song.
        """
        fn_with_ext = self.TOKEN_FILENAME + '.' + self.SAVE_FILETYPE
        return os.path.join(self.song_lyric_dir(songname), fn_with_ext)

    def song_embeddings_filename(self, songname):
        """Returns the full path of a particular song's embedding file

        Args:
            songname (str): The songname to find the directory for.

        Returns:
            str: The path of the file containing embedding data for the song.
        """
        fn_with_ext = self.LYRIC_EMBEDDING_FILENAME + '.' + self.SAVE_FILETYPE
        return os.path.join(self.song_embeddings_dir(songname),
                            fn_with_ext)

    def model_loc(self, model_name):
        """Returns the full path of the model with a particular name.

        Args:
            model_name (str): the name of the model (with extension)

        Returns:
            str: The full path of the model
        """
        return os.path.join(self.MODELS_PATH, model_name)

    def image_class_location(self):
        """Returns the location of the image classes file

        Returns:
            str: A full path to the file with the image classes
        """
        fn_with_ext = self.IMAGE_CLASS_FILENAME + '.' + self.SAVE_FILETYPE
        return os.path.join(self.DATA_DIR, fn_with_ext)

    def class_embeddings_filename(self):
        """Returns the location of the class embeddings file

        Returns:
            str: full path to the class embeddings file
        """
        fn_with_ext = self.CATEGORY_EMBEDDING_FILENAME + '.'\
            + self.SAVE_FILETYPE
        return os.path.join(self.DATA_DIR, fn_with_ext)

    def class_token_filename(self):
        """Returns the location of the tokenized classes for ImageNet

        Returns:
            [type]: [description]
        """
        fn_with_ext = self.CATEGORY_TOKEN_FILENAME + '.' + self.SAVE_FILETYPE
        return os.path.join(self.DATA_DIR, fn_with_ext)

    def complete_lyrics_filename(self, songname):
        """[summary]

        Args:
            songname ([type]): [description]

        Returns:
            [type]: [description]
        """
        fn_with_ext = self.FULL_LYRIC_FILENAME + '.' + self.SAVE_FILETYPE
        return os.path.join(self.song_lyric_dir(songname), fn_with_ext)

    def find_lrc_file(self, songname):
        songname_dir = self.song_lyric_dir(songname)
        logger.debug('Searching for lrc file.')

        logger.debug(
            f'Searching {self.LYRIC_PATH} for directory {songname}')
        if os.path.exists(songname_dir):
            logger.debug(
                f'Found directory {songname} in lyric directory: '
                f'{self.LYRIC_PATH}')
        else:
            logger.error(
                f'No such song directory {songname} in {self.LYRIC_PATH}. '
                'Have you created the directory?', exc_info=True)
            raise ValueError('Incorrect songname passed.')

        logger.debug(
            f'Looping through files in {songname_dir} to find lrc file')

        lrc_file = find_first_file_with_ext(songname_dir, 'lrc')

        return lrc_file

    def read_lrc_file(self, songname):
        logger.info(f'Reading lrc file for song {songname}')
        lrc_file = self.find_lrc_file(songname)
        try:
            with open(lrc_file, 'r') as f:
                ret = f.read()
        except TypeError:
            logger.error(
                f'Could not find a lrc file in the {songname} directory. Have you created it?', exc_info=True)
            raise
        else:
            logger.info(f'Succesfully read lrc file for {songname}.')

        return ret

    def read_id_to_img_class(self):
        loc = self.image_class_location()
        logger.debug(f'Loading image class yaml file from {loc}')
        try:
            with open(loc, 'r') as f:
                ret = yaml.load(f)
        except FileNotFoundError:
            logger.error(
                f'Could not find class yaml file at {loc}. Is the location correctly specified?')
            raise
        else:
            logger.debug(f'Succesfully loaded image to classes yaml file.')
        return ret

    @abstractmethod
    def word_embedder(self):
        pass

    @abstractmethod
    def gan_network(self):
        pass


class WikipediaBigGANGenerationEnviornment(GenerationEnvironment):

    def word_embedder(self):
        loc = self.model_loc('enwiki_20180420_100d.pkl')
        self.wordvec_dim = 100
        logger.info(f'Loading Wikipedia2Vec word embeddings model from {loc}.')
        model = Wikipedia2Vec.load(loc)
        return model

    def gan_network(self, resolution):
        logger.info(f'Loading BigGAN with resolution {resolution}.')
        model = BigGAN.from_pretrained(f'biggan-deep-{resolution}')
        return model
