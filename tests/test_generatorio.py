from deep_lyric_visualizer.generator.generatorio import GeneratorIO
from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)
from deep_lyric_visualizer.generator.generator_object import (GeneratorObject)

from unittest.mock import Mock, patch


class TestGeneratorIO:
    @patch.multiple(GeneratorIO, __abstractmethods__=set())
    def test_initialization(self):
        fake_gen_obj = Mock(GeneratorObject)
        fake_gen_obj.env = Mock(GenerationEnvironment)
        genio = GeneratorIO(fake_gen_obj)

        assert genio.env == fake_gen_obj.env
        assert isinstance(genio.env, GenerationEnvironment)
        assert genio.word_embedder == fake_gen_obj.env.word_embedder
        assert len(genio.word_to_vec) == 0
        assert genio.obj == fake_gen_obj
