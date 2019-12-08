.PHONY: env
env:
	conda env create -f environment.yml
	conda activate msnag
	conda uninstall pytorch -y  # In case it intalled the normal pytorch somehow
	conda install -c saareliad pytorch -y
	python -c"import torch"
