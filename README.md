# WIMU - music classification using few shot learning

WiMU-Gorski-Okrupa is a project designed to [provide a brief description of the project].

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have Poetry 1.7.0 installed.

1) Install poetry 1.7.0
```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
```

2) To install dependencies, run 
```bash 
make requirements 
```
- If you want to install pytorch with CUDA support, replace {version = "^2.1.1"} in pyproject.toml with {version = "^2.1.1", "source": "pytorch"}.
3) To download data first setup environment variables ACCESS_KEY_ID and SECRET_ACCESS_KEY or directly update the Makefile, then run `make download_data`
4) To start data loading, run `make dataset`.


Environment variables:
- SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
    - necessary for installing music-fsl as it depends on a depraceted version of scikit-learn.
- ACCESS_KEY_ID - access key to access the S3 bucket.
- SECRET_ACCESS_KEY - secret key to access the S3 bucket.


A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
