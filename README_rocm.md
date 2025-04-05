# Choose a new workload
With Pytorch_2_3_0_Rocm6_2_0 (ROCm 6.2.0 in the docker image)

# Connecting to workload
Once workload is in "running" state, use "interactive" credentials for SSH from workload. 
VS Code to this works stably. 
You can also circumvent the mandatory password entry by using ssh-copy-id, but you'll need to chmod permissions on ~/.ssh folder

# Update to get screen
sudo apt update
sudo apt install screen

# ENV var issues
Remove conda py_3.10 from PATH in /etc/bash.bashrc file 
# Creating new environments
Do not create new conda environments directly. These will be unstable, and will not work with the ROCm stack. Instead, use the conda env (py_3.10) in the docker image. This saves a lot of time and effort trying to get it to work (flash attention for example will fail without this).

### Clone off py_3.10 as a base for any new environment
conda create --name prism --clone py_3.10
conda create --name mllava --clone py_3.10

#### *CAREFUL* For installing libraries (such as torch/numpy/..even matplotlib)
Use conda install -n prism <library> instead of pip install <library>.
This is because conda will install the correct version of the library that is compatible with the ROCm stack.

For example: `pip install peft` worked just fine. `pip install matplotlib` completely messed up the environment, since it updates numpy. Doesn't work, screws up their carefully configured libraries. Bad idea! 
So, one *must* do `conda install` for those instead (`matplotlib` in this case).

#### For install prismatic
Ensure toml doesn't have specific versions. Do not reinstall torch, torchvision. You don't need torchaudio (Remove them from toml).
Let others install without version constraints. 

#### For installling using pip install -e . : use the following:
pip install --no-cache-dir -e .

#### TO install bitsandbytes:

##### If ROCm == 6.2.0: 

pip install --no-deps --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'

##### Else:

# Clone the github repo
git clone --recurse https://github.com/ROCm/bitsandbytes.git
cd bitsandbytes
git checkout rocm_enabled_multi_backend

# Install dependencies
pip install -r requirements-dev.txt

# Use -DBNB_ROCM_ARCH to specify target GPU arch
cmake -DBNB_ROCM_ARCH="gfx942" -DCOMPUTE_BACKEND=hip -S .

# Compile the project
make

# Install
python setup.py install