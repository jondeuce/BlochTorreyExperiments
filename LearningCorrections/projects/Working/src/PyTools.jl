module PyTools

using ..PyCall
import ..PyPlot
import ..Conda

####
#### Initialization
####

const torch = PyNULL()
const wandb = PyNULL()
const ignite = PyNULL()
const logging = PyNULL()

const plt = PyPlot
rcParams = PyDict()

function __pyinit__()
    copy!(torch, pyimport("torch"))
    copy!(wandb, pyimport("wandb"))
    copy!(ignite, pyimport("ignite"))
    copy!(logging, pyimport("logging"))

    py"""
    from ignite.contrib.handlers.wandb_logger import *
    """

    global rcParams = PyDict(PyPlot.matplotlib."rcParams")
    rcParams["text.usetex"] = false # use matplotlib internal tex rendering
    rcParams["mathtext.fontset"] = "cm" # "stix"
    rcParams["font.family"] = "cmu serif" # "STIXGeneral"
    rcParams["font.size"] = 12
    rcParams["axes.titlesize"] = "medium"
    rcParams["axes.labelsize"] = "medium"
    rcParams["xtick.labelsize"] = "small"
    rcParams["ytick.labelsize"] = "small"
    rcParams["legend.fontsize"] = "small"
end

function __init__()
    try
        __pyinit__()
    catch e
        if e isa PyCall.PyError
            @info "Installing python utilities..."
            __pyinstall__()
            __pyinit__()
        else
            rethrow(e)
        end
    end
end

####
#### Utils
####

"""
Converting between row major PyTorch `Tensor`s to Julia major Julia `Array`s
"""
p2j_array(x::PyObject) = _py_reverse_dims(x.detach().cpu().numpy()) # `Tensor` --> `Array`
j2p_array(x::AbstractArray) = torch.Tensor(_py_reverse_dims(x)) #  # `Array` --> `Tensor`
_py_reverse_dims(x::AbstractArray{T,N}) where {T,N} = permutedims(x, ntuple(i -> N-i+1, N))

####
#### Installation
####

function __pyinstall__()
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
    if !Sys.iswindows()
        run(`$(joinpath(Conda.ROOTENV, "bin", "pip")) install wandb`)
    else
        run(`$(joinpath(Conda.ROOTENV, "Scripts", "pip.exe")) install wandb`)
    end

    # Install hydra via pip (https://hydra.cc/docs/intro#installation)
    #   pip install hydra-core --upgrade
    # run(`$(joinpath(Conda.ROOTENV, "bin", "pip")) install hydra-core --upgrade`)
end

end # module
