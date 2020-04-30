import pytest

from deep_lyric_visualizer.generator.generator_object import (GeneratorObject)
from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)

from deep_lyric_visualizer.generator.generatorio import GeneratorIO, PickleGeneratorIO, YAMLGeneratorIO
from unittest.mock import Mock


class TestGenerator:

    def test_initialization(self):
        mock_pickle_wiki = Mock(WikipediaBigGANGenerationEnviornment)
        mock_yaml_wiki = Mock(WikipediaBigGANGenerationEnviornment)

        mock_pickle_wiki.SAVE_FILETYPE = 'pickle'
        mock_yaml_wiki.SAVE_FILETYPE = 'yaml'

        mock_pickle_some_other_gen_env = Mock(GenerationEnvironment)
        mock_yaml_some_other_gen_env = Mock(GenerationEnvironment)

        mock_yaml_some_other_gen_env.SAVE_FILETYPE = 'yaml'
        mock_pickle_some_other_gen_env.SAVE_FILETYPE = 'pickle'

        not_a_gen_env = Mock(str)

        gen = GeneratorObject(mock_pickle_wiki)
        assert isinstance(gen.env, GenerationEnvironment)
        assert isinstance(gen.env, WikipediaBigGANGenerationEnviornment)
        assert isinstance(gen.genio, GeneratorIO)
        assert isinstance(gen.genio, PickleGeneratorIO)

        gen = GeneratorObject(mock_yaml_wiki)
        assert isinstance(gen.env, GenerationEnvironment)
        assert isinstance(gen.env, WikipediaBigGANGenerationEnviornment)
        assert isinstance(gen.genio, GeneratorIO)
        assert isinstance(gen.genio, YAMLGeneratorIO)

        gen = GeneratorObject(mock_yaml_some_other_gen_env)
        assert isinstance(gen.env, GenerationEnvironment)
        assert not isinstance(gen.env, WikipediaBigGANGenerationEnviornment)
        assert isinstance(gen.genio, GeneratorIO)
        assert isinstance(gen.genio, YAMLGeneratorIO)

        gen = GeneratorObject(mock_pickle_some_other_gen_env)
        assert isinstance(gen.env, GenerationEnvironment)
        assert not isinstance(gen.env, WikipediaBigGANGenerationEnviornment)
        assert isinstance(gen.genio, GeneratorIO)
        assert isinstance(gen.genio, PickleGeneratorIO)

        with pytest.raises(ValueError):
            gen = GeneratorObject(not_a_gen_env)
