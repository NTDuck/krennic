.PHONY: install

install:
	pip install -r requirements.txt

format:
	python -m black ./krennic

lint:
	python -m flake8 ./krennic

test:
	python -m pytest ./tests
