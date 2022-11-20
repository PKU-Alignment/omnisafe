conda create -n omnisafe python=3.8
pip install -e .
pip install psutil
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

>  issue 1
pip install mpi4py

pip install gym[mujoco]
pip install scipy
pip install joblib
pip install tensorboard
pip install pyyaml
pip install tensorboardX
pip install cpprb
