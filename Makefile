print-%  : ; @echo $* = $($*)
PROJECT_NAME   = omnisafe
COPYRIGHT      = "OmniSafe Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) examples tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=

.PHONY: default
default: install

install:
	$(PYTHON) -m pip install -vvv .

install-editable:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools
	$(PYTHON) -m pip install -vvv --editable .

install-e: install-editable  # alias

uninstall:
	$(PYTHON) -m pip uninstall -y $(PROJECT_NAME)

build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

# Tools Installation

check_pip_install = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(1) --upgrade)
check_pip_install_extra = $(PYTHON) -m pip show $(1) &>/dev/null || (cd && $(PYTHON) -m pip install $(2) --upgrade)

pylint-install:
	$(call check_pip_install_extra,pylint,pylint[spelling])

flake8-install:
	$(call check_pip_install,flake8)
	$(call check_pip_install_extra,flake8-bugbear,flake8-bugbear)

py-format-install:
	$(call check_pip_install,isort)
	$(call check_pip_install,black)

mypy-install:
	$(call check_pip_install,mypy)

pre-commit-install:
	$(call check_pip_install,pre-commit)
	$(PYTHON) -m pre_commit install --install-hooks

docs-install:
	$(call check_pip_install_extra,pydocstyle,pydocstyle[toml])
	$(call check_pip_install_extra,doc8,"doc8<1.0.0a0")
	$(call check_pip_install,sphinx)
	$(call check_pip_install,sphinx-autoapi)
	$(call check_pip_install,sphinx-autobuild)
	$(call check_pip_install,sphinx-copybutton)
	$(call check_pip_install,sphinx-autodoc-typehints)
	$(call check_pip_install_extra,sphinxcontrib.spelling,sphinxcontrib.spelling pyenchant)

pytest-install:
	$(call check_pip_install,pytest)
	$(call check_pip_install,pytest-cov)
	$(call check_pip_install,pytest-xdist)

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

# Tests

pytest: pytest-install
	cd tests && $(PYTHON) -c 'import $(PROJECT_NAME)' && \
	$(PYTHON) -m pytest --verbose --color=yes --durations=0 \
		--cov="$(PROJECT_NAME)" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) .

test: pytest

# Python linters

pylint: pylint-install
	$(PYTHON) -m pylint $(PROJECT_PATH)

flake8: flake8-install
	$(PYTHON) -m flake8 $(PYTHON_FILES) --count --select=E9,F63,F7,F82,E225,E251 --show-source --statistics

py-format: py-format-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) --check $(PYTHON_FILES) && \
	$(PYTHON) -m black --check $(PYTHON_FILES)

mypy: mypy-install
	$(PYTHON) -m mypy $(PROJECT_PATH)

pre-commit: pre-commit-install
	$(PYTHON) -m pre_commit run --all-files

# Documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") -check $(SOURCE_FOLDERS)

docstyle: docs-install
	make -C docs clean
	$(PYTHON) -m pydocstyle $(PROJECT_PATH) && doc8 docs && make -C docs html SPHINXOPTS="-W"

docs: docs-install
	$(PYTHON) -m sphinx_autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	make -C docs clean
	make -C docs spelling SPHINXOPTS="-W"

clean-docs:
	make -C docs clean

# Utility functions

# TODO: add mypy when ready
# TODO: add docstyle when ready
lint: flake8 py-format pylint addlicense spelling

format: py-format-install addlicense-install
	$(PYTHON) -m isort --project $(PROJECT_NAME) $(PYTHON_FILES)
	$(PYTHON) -m black $(PYTHON_FILES)
	addlicense -c $(COPYRIGHT) -ignore tests/coverage.xml -l apache -y 2022-$(shell date +"%Y") $(SOURCE_FOLDERS)

clean-py:
	find . -type f -name  '*.py[co]' -delete
	find . -depth -type d -name "__pycache__" -exec rm -r "{}" +
	find . -depth -type d -name ".mypy_cache" -exec rm -r "{}" +
	find . -depth -type d -name ".pytest_cache" -exec rm -r "{}" +
	rm tests/.coverage
	rm tests/coverage.xml

clean-build:
	rm -rf build/ dist/
	rm -rf *.egg-info .eggs envs/safety-gymnasium/*.egg-info envs/safety-gymnasium/.eggs

clean: clean-py clean-build clean-docs
