import Conda

# Install pip into conda environment
Conda.add("pip")

# Install pytorch for the cpu via conda (https://pytorch.org/get-started/locally/):
#   conda install pytorch torchvision cpuonly -c pytorch
Conda.add(["pytorch", "torchvision", "cpuonly"]; channel = "pytorch")

# Install pytorch ignite via conda (https://github.com/pytorch/ignite#installation):
#   conda install ignite -c pytorch
Conda.add("ignite"; channel = "pytorch")

# Install wandb via pip (https://docs.wandb.com/quickstart)
#   pip install wandb
run(`$(joinpath(Conda.ROOTENV, "bin", "pip")) install wandb`)

# Install hydra via pip (https://hydra.cc/docs/intro#installation)
#   pip install hydra-core --upgrade
run(`$(joinpath(Conda.ROOTENV, "bin", "pip")) install hydra-core --upgrade`)
