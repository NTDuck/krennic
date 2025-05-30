.PHONY: install

install:
	pip install -r requirements.txt

format:
	python -m black ./polynomial-regression

lint:
	python -m flake8 ./polynomial-regression

test:
	python -m pytest ./tests
