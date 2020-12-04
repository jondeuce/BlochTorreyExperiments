module PyTools

using ..Reexport
@reexport using PyCall

import PyPlot, Conda
export PyPlot, Conda

#### PyPlot

const plt = PyPlot
const rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = false

export plt, rcParams

#### ML Tools: Torch, WandB, Ignite, etc.

const torch = PyNULL()
const wandb = PyNULL()
const ignite = PyNULL()
const logging = PyNULL()

export torch, wandb, ignite, logging

function __init__()
    copy!(torch, pyimport("torch"))
    copy!(wandb, pyimport("wandb"))
    copy!(ignite, pyimport("ignite"))
    copy!(logging, pyimport("logging"))

    py"""
    from ignite.contrib.handlers.wandb_logger import *
    """
end

#### Python helpers

# Converting between row major PyTorch `Tensor`s to Julia major Julia `Array`s
reversedims(x::AbstractArray{T,N}) where {T,N} = permutedims(x, ntuple(i -> N-i+1, N))
array(x::PyObject) = reversedims(x.detach().cpu().numpy()) # `Tensor` --> `Array`
array(x::AbstractArray) = torch.Tensor(reversedims(x)) #  # `Array` --> `Tensor`

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
