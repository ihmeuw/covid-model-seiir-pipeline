# makefile for easy manage package
.PHONY: clean

ENV_NAME=test_runs_20200514
CONDA_PREFIX=/ihme/homes/nathenry/miniconda3/

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


install_env:
	( \
		source $(CONDA_PREFIX)/etc/profile.d/conda.sh && \
		conda create -n $(ENV_NAME) python=3.7 && \
		conda activate $(ENV_NAME) && \
		pip install numpy scipy pandas matplotlib pyyaml pytest xspline jobmon && \
		conda install -c conda-forge cyipopt && \
		git clone https://github.com/zhengp0/limetr.git && \
		cd limetr && make install && cd .. && \
		git clone https://github.com/ihmeuw-msca/MRTool.git && \
		cd MRTool && git checkout seiir_model && python setup.py install && cd .. && \
		git clone https://github.com/zhengp0/SLIME.git && \
		cd SLIME && python setup.py install && cd .. && \
		git clone https://github.com/ihmeuw-msca/ODEOPT.git && \
		cd ODEOPT && git checkout random && python setup.py install && cd .. && \
		git clone https://github.com/ihmeuw-msca/SEIRPipeline.git && \
		cd SEIRPipeline && git checkout old &&python setup.py install && cd .. && \
		python setup.py install; \
    )

install_env_ssh:
        ( \
                source $(CONDA_PREFIX)/etc/profile.d/conda.sh && \
                conda create -n $(ENV_NAME) python=3.7 && \
                conda activate $(ENV_NAME) && \
                pip install numpy scipy pandas matplotlib pyyaml pytest xspline jobmon && \
                conda install -c conda-forge cyipopt && \
                git clone git@github.com:zhengp0/limetr.git && \
                cd limetr && make install && cd .. && \
                git clone git@github.com:ihmeuw-msca/MRTool.git && \
                cd MRTool && git checkout seiir_model && python setup.py install && cd .. && \
                git clone git@github.com:zhengp0/SLIME.git && \
                cd SLIME && python setup.py install && cd .. && \
                git clone git@github.com:ihmeuw-msca/ODEOPT.git && \
                cd ODEOPT && git checkout random && python setup.py install && cd .. && \
                git clone git@github.com:ihmeuw-msca/SEIRPipeline.git && \
                cd SEIRPipeline && git checkout old &&python setup.py install && cd .. && \
                python setup.py install; \
    )

uninstall_env:
	conda remove --name $(ENV_NAME) --all
