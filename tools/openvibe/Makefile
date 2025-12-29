VENV?=.venv
PYTHON?=python3

.PHONY: setup run fmt clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

run: setup
	$(VENV)/bin/python -m openvibe.core sample_data.csv --plot --stdout-report

serve: setup
	$(VENV)/bin/uvicorn service:app --reload

fmt:
	$(VENV)/bin/python -m black openvibe.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__

