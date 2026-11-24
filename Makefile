.PHONY: install data train train-quick test run docker-build docker-run

install:
	pip install -r requirements-dev.txt

data:
	python -m src.data --n 15000 --out data/loans.csv

train:            ## full grid search, 5-fold stratified CV (~3 min on 2 cores)
	python -m src.train

train-quick:      ## tiny grids, ~10 s
	python -m src.train --quick

test:
	python -m pytest

run:              ## development server on :8000
	flask --app src.wsgi run --port 8000

docker-build:     ## requires models/ from `make train`
	docker build -t loan-default-api .

docker-run:
	docker run --rm -p 8000:8000 loan-default-api
