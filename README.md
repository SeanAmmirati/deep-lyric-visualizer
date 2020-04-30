# Deep Lyric Visualization Generator

An enhancement to a [GAN music visualizer](https://github.com/msieg/deep-music-visualizer) that uses contextual lyric information
in a song to select categories for visualization.

The result is something like this:

 [![YouTube video example.](https://img.youtube.com/vi/kkpWfGzoems/1.jpg)](https://www.youtube.com/watch?v=kkpWfGzoems)

## Description

This project is the start of some ideas surrounding trying to build upon the
 music visualization using GAN from [this repository](https://github.com/msieg/deep-music-visualizer) and adding a lyrical element.

To achieve this, this code vectorizes the lyrics to the song and compares them
to the ImageNet categories. Then, when running the GAN, it will use the categories
most similar to those in current lyrics as evaluated by the simple NLP model.

Similarity is considered at the word, line, and stanza level. This is a rough
assessment, and can be tweaked.

At the moment, some of the code is a bit all over the place. Apologies for this
and any confusion. I will be updating this code in the coming weeks/months
to make it more legible and useable.

If you'd like to contribute, know more, or give any suggestions, please feel
free to reach out.

## Pre-Requisites


In order to use this repository, you must first do a few things.

1. **Select an appropriate word embedding model.** You can train your own model, or
used a pretrained model. This project assumes use of the [Wikipedia2Vec model.](https://wikipedia2vec.github.io/wikipedia2vec).

2. **Select an appropriate model to fit the General Adverserial Network (GAN).**
Again, you can train your own model or use a pretrained model. This project
 assumes use of [BigGAN.](https://arxiv.org/abs/1809.11096)

3. **Set up the mp3 file and the .lrc file for the song.** You must have an .mp3
file for the song that you wish to create a visualization for, as well as a
.lrc file for the lyrics. To generate one of these yourself, try using
[this website](https://lrcgenerator.com/) with lyrics sourced from the common
lyric sources -- Genius, AZLyrics, etc.

### Embedder

#### Using the default embedder

If you would like to proceed using the defaults, please download the correct
[pretrained model](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).
The correct pretrained model at this time is enwiki_20180420_100d.pkl.bz2, or
the binary file for 100 dimensions. Save this file to PROJECT_ROOT/models.

#### Using another Wikipedia2Vec embedder

Any of the [pretrained models](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) can be used with this package quite easily.
If you would like to use a different Wikipedia2Vec model, look in the
`PROJECT_ROOT/src/config directory` for the `default_cfg.yaml` file.

You should see an entry like this:

```yaml
WIKIPEDIA_2_VEC_MODEL_NAME: enwiki_20180420_100d.pkl.bz2
```

#### Using another embedder entirely

If you would like to use another embedder, you should create a subclass of the
`GenerationEnvironment` class in `generation_environment.py`. All of the
objects in this repository (so called `GeneratorObjects`) take the
`GenerationEnvironment` as a parameter.

Simply add a method named `word_embedder` and have this method return the model.
This was done to make the process flexible.

Note that you will need to ensure that the API for the loaded classes matches
that of Wikipedia2Vec for this to work out of the box for methods used by the
vectorizer class. As of this writing, the only such method is the
`get_word_vector` method, which should be straight-forward to port from
other models.

### GAN

#### Using the default GAN

There is no setup necessary to use the default GAN, as this will be loaded by
BigGAN.

#### Using a custom GAN

If you would like to use another GAN, you should create a subclass of the
`GenerationEnvironment` class in `generation_environment.py`. All of the
objects in this repository (so called `GeneratorObjects`) take the
`GenerationEnvironment` as a parameter.

Simply add a method named `gan_network` and have this method return the model.
This was done to make the process flexible.

Currently, this is used only in the `deep-music-visualizer` portion of the
project. The API must match that of `deep-music-visualizer`.

## Project Organization
```
├── LICENSE
├── Makefile                               <- Makefile with commands like `make data` or `make train` (Generation is in process, please be patient...)
├── README.md                              <- The top-level README for developers using this project.
├── data
│   ├── embeddings                         <- Storage for the word embeddings for particular songs.
│   ├── lyrics                             <- Where lyric folders are located. Please create a new folder here with the associated .lrc file to use this. More info in future iterations.
│   └── processed                          <- Processed vectorizations/other data that is timeconsuming to generate and will be consistent across runs.
│
├── docs                                   <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                                 <- Word vectorizers that were used (pre-trained). WikiToVec was used for this iteration.
│
├── references                             <- Data dictionaries, manuals, and all other explanatory materials.
│
├── examples                               <- I stored .mp4 examples here for later use.
│
├── requirements.txt                       <- The requirements file for reproducing the analysis environment, e.g.
│                                             generated with `pip freeze > requirements.txt` Look also for the environment.yaml
│
├── setup.py                               <- makes project pip installable (pip install -e .) so src can be imported
├── src                                    <- Source code for use in this project.
│   ├── __init__.py                        <- Makes src a Python module
│   │
│   ├── deep-music-visualizer              <- The original repo that this is inspired from. Tweaks were made to the vizualize.py function to integrate lyrical embeddings.
│   │
│   ├── generator                          <- Inheritable classes representing basic behaviors for the remainder of the classes. This contains information about where files should be loaded
│   │   │                                     from and saved to for a particular class.
│   │   │
│   │   ├─ generation_object.py            <- A generation object contains all of the information needed to generate from the environment. This is the most generic object of this project, and most
│   │   │                                    other classes inherit from it (or are being altered to inherit from it).
│   │   │
│   │   ├── generation_environment.py      <- An environment class that handles anything relating to the directory structure and environment which the code is being run in. Contains the abstract class
│   │   │                                     which can be used to use custom word embeddors and gan_networks. The default used for this project was Wikipedia2Vec and BigGAN but others can be specified.
│   │   │                                     This class also handles the loading of external files.
│   │   │
│   │   │
│   │   └─ generatorio.py                  <- GeneratorIO classes handle the saving and loading of the objects themselves for later use.
│   │
│   ├── image_categories                   <- Classes which handle the NLP surrounding the ImageNet categories.
│   │   │
│   │   ├── image_categories.py            <- Overall ImageCategory class. Holds the Vectorizer and Tokenizer within it.
│   │   │
│   │   ├── image_category_tokenizer.py    <- Handles Tokenization of the ImageNet categories.
│   │   │
│   │   └── image_category_vectorizer.py   <- Handles Vectorization of the ImageNet categories.
│   │
│   ├── lyrics                             <- Classes which handle the NLP and weighings surrounding the actual lyrics.
│   │   │
│   │   ├── lyrics.py                      <- Overall Lyrics class. Holds all objects related to the manipulation of lyrics.
│   │   │
│   │   ├── lyric_tokenizer.py             <- Handles Tokenization of the lyrics.
│   │   │
│   │   ├── lyric_vectorizer.py            <- Handles Vectorization of the lyrics.
│   │   │
│   │   └── lyric_weigher.py               <- Handles weighing of the lyrics (within a line, stanza, etc).
│   │
│   ├── nlp                                <- Classes which handle NLP operations used throughout -- i.e. for both Lyrics and ImageNet. Can be expanded to other usecases.
│   │   │
│   │   ├── tokenizer.py                   <- A general tokenizer class that performs the necessary tokenizations. Inherited in lyrics/image_categories.
│   │   │
│   │   ├── vectorizer.py                  <- A general vectorizer class that performs the necessary tokenizations. Inherited in lyrics/image_categories.
│   │   │
│   │   └── wordvector_similarity.py       <- A helper class to quantify wordvector similarity, to be used for comparisons elsewhere throughout the code.
│   │
│   └── helpers.py                         <- Generic helper functions useful in various usecases.
│
└── tox.ini                                <- tox file with settings for running tox; see tox.testrun.org
```
