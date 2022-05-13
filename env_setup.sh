conda create -n detectron2_env python==3.9.1 -y
source activate detectron2_env
pip install numpy pandas pynrrd pillow opencv-python ipython scikit-image jedi==0.17.2
pip install --verbose "https://download.pytorch.org/whl/cpu/torch-1.9.0%2Bcpu-cp39-cp39-linux_x86_64.whl"
pip install --verbose https://download.pytorch.org/whl/cpu/torchvision-0.10.0%2Bcpu-cp39-cp39-linux_x86_64.whl

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
