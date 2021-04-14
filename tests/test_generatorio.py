from deep_lyric_visualizer.generator.generatorio import GeneratorIO
from deep_lyric_visualizer.generator.generation_environment import (GenerationEnvironment,
                                                                    WikipediaBigGANGenerationEnviornment)
from deep_lyric_visualizer.generator.generator_object import (GeneratorObject)

from unittest.mock import Mock, patch
import deep_lyric_visualizer.generator.generatorio


class TestGeneratorIO:
    @patch.multiple(GeneratorIO, __abstractmethods__=set())
    def test_initialization(self):
        fake_gen_obj = Mock(GeneratorObject)
        fake_gen_obj.env = Mock(GenerationEnvironment)
        genio = GeneratorIO(fake_gen_obj)

        assert genio.env == fake_gen_obj.env
        assert isinstance(genio.env, GenerationEnvironment)
        assert genio.word_embedder == fake_gen_obj.env.word_embedder()
        assert len(genio.word_to_vec) == 0
        assert genio.obj == fake_gen_obj

    @patch.multiple(GeneratorIO, __abstractmethods__=set())
    def test_get_saving_location(self):
        generic_env_mock = Mock(GenerationEnvironment)

        generic_object_mock = Mock(GeneratorObject)

        generic_object_mock.env = generic_env_mock
        generic_object_mock.name = 'lyric_tokenizer'

        genio = GeneratorIO(generic_object_mock)
        ret = genio.get_saving_location('anything')
        assert generic_env_mock.song_lyric_filename.called_once()
        assert genio.save_loc == generic_env_mock.song_lyric_filename(
            'anything')
        assert genio.save_loc == ret

        generic_object_mock.name = 'lyric_vectorizer'
        genio = GeneratorIO(generic_object_mock)
        ret = genio.get_saving_location('anything')
        assert generic_env_mock.song_embeddings_filename.called_once()
        assert genio.save_loc == generic_env_mock.song_embeddings_filename(
            'anything')
        assert genio.save_loc == ret

        generic_object_mock.name = 'image_category_vectorizer'
        genio = GeneratorIO(generic_object_mock)
        ret = genio.get_saving_location('anything')
        assert generic_env_mock.class_embeddings_filename.called_once()
        assert genio.save_loc == generic_env_mock.class_embeddings_filename()
        assert genio.save_loc == ret

        generic_object_mock.name = 'image_category_tokenizer'
        genio = GeneratorIO(generic_object_mock)
        ret = genio.get_saving_location('anything')
        assert generic_env_mock.class_token_filename.called_once()
        assert genio.save_loc == generic_env_mock.class_token_filename()
        assert genio.save_loc == ret

        generic_object_mock.name = 'lyrics'
        genio = GeneratorIO(generic_object_mock)
        ret = genio.get_saving_location('anything')
        assert generic_env_mock.complete_lyrics_filename.called_once()
        assert genio.save_loc == generic_env_mock.complete_lyrics_filename(
            'anything')
        assert genio.save_loc == ret

    @patch.multiple(GeneratorIO, __abstractmethods__=set())
    @patch('deep_lyric_visualizer.generator.generatorio.os')
    def test_make_save_locations(self, os_mock):
        generic_env_mock = Mock(GenerationEnvironment)

        os_mock.path.exists = lambda x: True
        os_mock.mkdir = lambda x: True

        generic_object_mock = Mock(GeneratorObject)
        generic_object_mock.env = generic_env_mock
        generic_object_mock.name = 'lyric_tokenizer'

        genio = GeneratorIO(generic_object_mock)
        genio.make_save_locations('test')
