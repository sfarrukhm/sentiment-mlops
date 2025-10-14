install:
	pip install --upgrade pip &&\
	pip install --requirements.txt

test:
	pytest -v

format:
	black **/*.py


lint:
	pylint --disable=R,C, app/serve.py

all: install lint format

		

