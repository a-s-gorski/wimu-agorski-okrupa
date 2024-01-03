.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = wimu-gorski-okrupa
PYTHON_INTERPRETER = python3
	
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel poetry==1.7.0
	$(PYTHON_INTERPRETER) poetry shell
	$(PYTHON_INTERPRETER) poetry install

update_data:
	dvc add data/.
	dvc push -r origin

download_data:
	poetry shell
	dvc remote add origin s3://dvc -f
	if [ -z "$(ACCESS_KEY_ID)" ] || [ -z "$(SECRET_ACCESS_KEY)" ]; then \
		echo "Error: Both ACCESS_KEY_ID and SECRET_ACCESS_KEY environment variables are required."; \
		exit 1; \
	fi
	dvc remote modify origin endpointurl https://dagshub.com/a-s-gorski/datasets-wimu.s3
	dvc remote modify origin --local access_key_id $(ACCESS_KEY_ID)
	dvc remote modify origin --local secret_access_key $(SECRET_ACCESS_KEY)
	dvc pull -r origin -f

extract_data:
	unzip -o data/raw/tiny_sol/TinySOL.zip -d data/interim/tiny_sol
	unzip -o data/raw/good_sounds/good-sounds.zip -d data/interim/good_sounds
	unzip -o data/raw/irmas/IRMAS-TrainingData.zip -d data/interim/irmas
	unzip -o data/raw/kaggle_wavefile_of_instruments/kaggle_wavefiles_of_instruments.zip -d data/interim/kaggle_wavefiles_of_instruments

## Make Dataset
dataset:
	$(PYTHON_INTERPRETER) -m src.data.make_dataset irmas config/dataset.yml data/interim/irmas/IRMAS-TrainingData data/processed

## Train the model
train:
	$(PYTHON_INTERPRETER) -m src.models.train_model irmas config/training.yml data/processed data/model_output/irmas

format_code:
	cd src && isort . && autopep8 -i -r --max-line-length 79 -a -a -a  .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src



## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


