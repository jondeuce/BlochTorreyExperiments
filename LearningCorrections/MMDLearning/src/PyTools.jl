module PyTools

import ..Plots
import PyCall, PyPlot, Conda
export PyCall, PyPlot, Conda

#### PyPlot

const plt = PyPlot
const rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = false

export plt, rcParams

#### ML Tools: Torch, WandB, Ignite, etc.

const torch = PyCall.PyNULL()
const wandb = PyCall.PyNULL()
const ignite = PyCall.PyNULL()
const logging = PyCall.PyNULL()

export torch, wandb, ignite, logging

function __init__()
    copy!(torch, PyCall.pyimport("torch"))
    copy!(wandb, PyCall.pyimport("wandb"))
    copy!(ignite, PyCall.pyimport("ignite"))
    copy!(logging, PyCall.pyimport("logging"))

    PyCall.py"""
    from ignite.contrib.handlers.wandb_logger import *
    """
end

#### Python modules installation

function setup()
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
    # run(`$(joinpath(Conda.ROOTENV, "bin", "pip")) install hydra-core --upgrade`)
end

end # module
