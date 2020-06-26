# All commands to a target (e.g., install_env) execute within a single shell
#
# https://www.gnu.org/software/make/manual/html_node/One-Shell.html
.ONESHELL:
# use bash by default
SHELL := /bin/bash

# force make to run serially (basically disables the -j flag)
# this is used to ensure the env creation/activation/package installation happens in order
.NOTPARALLEL:

# pip picks up this env var; equivalent to --extra-index-url
export PIP_EXTRA_INDEX_URL = https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/

# makefile for easy manage package
ENV_NAME=dummy_TESTING

.PHONY: clean
clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "MANIFEST" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	rm -rf limetr MRTool ODEOPT SEIRPipeline SLIME


.PHONY: install_env
install_env: build_conda_env install_github_packages
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda activate $(ENV_NAME)
	@# NOTE: this previously checked out 'develop' - now the user has to already be on the right branch
	pip install -e .[dev]

.PHONY: build_conda_env
build_conda_env:
	source $(CONDA_PREFIX)/etc/profile.d/conda.sh
	conda create -n $(ENV_NAME) -y -c conda-forge cyipopt python=3.7


.PHONY: install_github_packages
install_github_packages:
	git clone git@github.com:zhengp0/limetr.git
	cd limetr && make install && cd ..
	git clone git@github.com:ihmeuw-msca/MRTool.git
	cd MRTool && git checkout seiir_model && python setup.py install && cd ..
	git clone git@github.com:zhengp0/SLIME.git
	cd SLIME && python setup.py install && cd ..


.PHONY: test
test:
	pytest tests


.PHONY: uninstall_env
uninstall_env:
	conda remove --name $(ENV_NAME) --all


.PHONY: travis_install_env
travis_install_env: install_github_packages_https
	pip install -e .


.PHONY: install_conda
install_conda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	@# install in batch mode (assumes you agree to license) at -p(refix) and-f(orce)
	@# NOTE: this does NOT add miniconda binaries to our path. we do that separately
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda -f


.PHONY: install_github_packages_https
# export PATH value. IMPORTANT - use := and not = so it is non-recursive
# https://stackoverflow.com/a/1605665
install_github_packages_https: export PATH:=$(shell pwd)/miniconda/bin:$(PATH)
install_github_packages_https: install_conda
	export PATH
	git clone https://github.com/zhengp0/limetr.git
	cd limetr && make install && cd ..
	git clone https://github.com/ihmeuw-msca/MRTool.git
	cd MRTool && git checkout seiir_model && python setup.py install && cd ..
	git clone https://github.com/zhengp0/SLIME.git
	cd SLIME && python setup.py install && cd ..
