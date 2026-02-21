install:
	pip install -r requirements.txt

test:
	pytest -v

format:
	black .

lint:
	flake8 src
