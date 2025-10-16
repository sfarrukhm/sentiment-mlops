install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -v

format:
	black **/*.py


lint:
	pylint --disable=R,C, app/serve.py

all: install lint test format

		

