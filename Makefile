.PHONY: install lint test download eda train all docker-build docker-run

install:
	pip install -r requirements.txt

lint:
	ruff check src scripts tests

test:
	pytest

download:
	python scripts/download_data.py --output data/raw/online_shoppers_intention.csv

eda:
	python -m src.eda --data-path data/raw/online_shoppers_intention.csv --output-dir report/images

train:
	python -m src.train --data-path data/raw/online_shoppers_intention.csv --output-dir artifacts --models-dir models --n-iter 12

all: download eda train

docker-build:
	docker compose build

docker-run:
	docker compose up --build
