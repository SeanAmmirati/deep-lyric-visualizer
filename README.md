Deep Lyric Visualization Generator
==============================

An enhancement to a [GAN music visualizer](https://github.com/msieg/deep-music-visualizer) that uses contextual lyric information
in a song to select categories for visualization.


The result is something like this:

 [![YouTube video example.](https://img.youtube.com/vi/kkpWfGzoems/0.jpg)](https://www.youtube.com/watch?v=kkpWfGzoems)

# Description

This project is the very start of some ideas surrounding trying to take some
things that have already been done in this space in terms of music visualization
using GAN (from [this repository](https://github.com/msieg/deep-music-visualizer)
and adding a lyrical element.

This is very simplistic at the moment, and was part of a side project for fun.
To achieve this, this code vectorizes the lyrics to the song and compares them
to the imagenet categories. Then, when running the GAN, it will use the categories
most similar to those in current lyrics as evaluated by the simple NLP model.

Similarity is considered at the word, line, and stanza level. This is a rough
assessment, and can be tweaked.

At the moment, some of the code is a bit all over the place. Apologies for this
and any confusion. I will be updating this code in the coming weeks/months
to make it more legible and useable.

If you'd like to contribute, know more, or give any suggestions, please feel
free to reach out.

Project Organization
------------

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
