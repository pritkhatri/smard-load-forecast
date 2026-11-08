PY ?= python

.PHONY: install data evaluate train serve test lint all

install:
	$(PY) -m pip install -e ".[dev,nn]"

data:      ## download Amprion hourly load from SMARD and validate it
	$(PY) -m smardfc data

evaluate:  ## cross-validation + held-out test, writes results/
	$(PY) -m smardfc evaluate

train:     ## fit the headline model on all data, writes models/model.joblib
	$(PY) -m smardfc train

serve:     ## start the API on http://127.0.0.1:8000
	$(PY) -m smardfc serve

test:
	$(PY) -m pytest

lint:
	ruff check . && ruff format --check .

all: data evaluate train
