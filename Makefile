.PHONY: env
env:
	# Install initial "easy" requirements from file.
	conda env create -f environment.yml
	conda activate msnag

	# Install pytorch:
	# optional, remove previous installation
	# conda uninstall pytorch -y  # In case it intalled the normal pytorch somehow
	conda install -c saareliad pytorch -y
	python -c"import torch"
	
	# Install torchvision:
	# Note: we need to do it after we installed pytorch.
	# (1) Install faster pollow-simd with AVX2 support.
	# can if we have AVX2 support with grep avx2 /proc/cpuinfo
	pip uninstall pillow
	CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
	# (2) Install torchvision from source.
	pip install git+https://github.com/pytorch/vision.git


