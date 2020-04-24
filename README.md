Deep Lyric Visualizer -- GANs
==============================

An enhancement to a [GAN music visualizer](https://github.com/msieg/deep-music-visualizer) that uses contextual lyric information
in a song to select categories for visualization.


The result is something like this:

 [![YouTube video example.](https://img.youtube.com/vi/kkpWfGzoems/0.jpg)](https://www.youtube.com/embed/kkpWfGzoems)

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
    ├── Makefile                           <- Makefile with commands like `make data` or `make train` (Generation is in process, please be patient...)
    ├── README.md                          <- The top-level README for developers using this project.
    ├── data
    │   ├── embeddings                     <- Storage for the word embeddings for particular songs.
    │   ├── lyrics                         <- Where lyric folders are located. Please create a new folder here with the associated .lrc file to use this. More info in future iterations.
    │   └── processed                      <- Processed vectorizations/other data that is timeconsuming to generate and will be consistent across runs.
    │
    ├── docs                               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                             <- Word vectorizers that were used (pre-trained). WikiToVec was used for this iteration.
    │
    ├── references                         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── examples                           <- I stored .mp4 examples here for later use.
    │
    ├── requirements.txt                   <- The requirements file for reproducing the analysis environment, e.g.
    │                                      generated with `pip freeze > requirements.txt` Currently not terribly
    │                                      useful because it has not yet been updated. Look for the environment.yaml
    │
    ├── setup.py                           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                                <- Source code for use in this project.
    │   ├── __init__.py                    <- Makes src a Python module
    │   │
    │   ├── deep-music-visualizer          <- The original repo that this is inspired from. Tweaks were made to the vizualize.py function to integrate lyrical embeddings.
    │   │
    │   ├── features                       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization                  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
