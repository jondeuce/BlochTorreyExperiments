module PyTools

using ..PyCall
import ..PyPlot
import ..Conda

export torch, wandb, ignite, logging
export plt, rcParams, wrap_catch_interrupt
export @j2p

#### Initialize python libraries

const torch = PyNULL()
const wandb = PyNULL()
const ignite = PyNULL()
const logging = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
    copy!(wandb, pyimport("wandb"))
    copy!(ignite, pyimport("ignite"))
    copy!(logging, pyimport("logging"))

    py"""
    from ignite.contrib.handlers.wandb_logger import *
    """
end

#### PyPlot

const plt = PyPlot
const rcParams = plt.PyDict(plt.matplotlib."rcParams")
rcParams["font.size"] = 10
rcParams["text.usetex"] = false

#### Python helpers

"""
Converting between row major PyTorch `Tensor`s to Julia major Julia `Array`s
"""
array(x::PyObject) = _py_reverse_dims(x.detach().cpu().numpy()) # `Tensor` --> `Array`
array(x::AbstractArray) = torch.Tensor(_py_reverse_dims(x)) #  # `Array` --> `Tensor`
_py_reverse_dims(x::AbstractArray{T,N}) where {T,N} = permutedims(x, ntuple(i -> N-i+1, N))

function wrap_catch_interrupt(f; msg = "")
    function wrap_catch_interrupt_inner(engine, args...)
        f(engine, args...)
        # try
        #     f(engine, args...)
        # catch e
        #     if e isa InterruptException
        #         @info "User interrupt"
        #     else
        #         !isempty(msg) && @warn msg
        #         @warn sprint(showerror, e, catch_backtrace())
        #     end
        #     engine.terminate()
        # end
    end
end

"""
Convert Julia callback function to Python function.
Julia functions can already by passed directly via pycall
but it's via a wrapper type that will error if Python tries
to inspect the Julia function too closely, e.g. counting
the number of arguments, etc.

First argument is assumed to be the `engine` object,
which is used to terminate training in an error or
user interrupt occurs.
"""
macro j2p(f)
    local wrapped_f = :(wrap_catch_interrupt($(esc(f))))
    local jlfun2pyfun = esc(:(PyCall.jlfun2pyfun))
    quote
        $jlfun2pyfun($wrapped_f)
    end
end

#### Python modules installation

function install()
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
