from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)
from deep_lyric_visualizer.generator.generatorio import PickleGeneratorIO, YAMLGeneratorIO
from deep_lyric_visualizer.helpers import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)


class GeneratorObject:

    def __init__(self, gen_env=None):
        """An abstract, generator object. This is the more general object
        of this project, which contains utilities and environment specifics
        that all classes can use.

        Args:
            gen_env ([GenerationEnvironment, optional): A generation
            environment object, which specifies the file structure and default
            models for the project.
            See generation_environment.py for more details.
            Defaults to None, which will initialize a WikipediaBigGAN
            environment.

        Raises:
            ValueError: This is raised if the passed object is not a valid
            GenerationEnvironment object.
        """
        logger.debug(
            f'Started initialization of {self.__class__.__name__}class.')

        if not gen_env:
            logger.info(
                'No passed environment class. Defaulting to Wikipedia2Vec and '
                'BigGAN.')

            gen_env = WikipediaBigGANGenerationEnviornment()

        else:
            if __name__ != '__main__':
                logger.debug('Passed environment class from other class')

            else:
                logger.info('Custom enviornment class passed.')

            if not isinstance(gen_env, GenerationEnvironment):
                logger.error(
                    'Argument gen_env is not a GenerationEnvironment instance.'
                    ' You must pass the appropriate object here.')
                raise ValueError('Not a Generation Environment instance.')

        self.env = gen_env

        if self.env.SAVE_FILETYPE == 'pickle':
            self.genio = PickleGeneratorIO(
                self, self.env)
        elif self.env.SAVE_FILETYPE == 'yaml':
            self.genio = YAMLGeneratorIO(self, self.env)
