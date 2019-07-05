# Initialize project/code loading
import Pkg
using Printf
using Statistics: mean, median, std
using StatsBase: quantile, sample, iqr
using Base.Iterators: repeated, partition
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))

using MWFLearning
using StatsPlots
pyplot(size=(800,600))

# Utils
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"
savebson(filename, data::Dict) = @elapsed BSON.bson(filename, data) # TODO getnow()
# savebson(filename, data::Dict) = @elapsed DrWatson.@tagsave(filename, data, false, gitdir()) # TODO getnow(), safe = false

# Settings
const settings_file = "settings.toml"
const settings = verify_settings(TOML.parsefile(settings_file))
const model_settings = settings["model"]

const FILE_PREFIX = getnow() * "." * DrWatson.savename(model_settings) * "."
const GPU = settings["gpu"] :: Bool
const T   = settings["prec"] == 32 ? Float32 : Float64
const VT  = Vector{T}
const MT  = Matrix{T}

const savefoldernames = ["settings", "models", "weights", "log", "plots"]
const savefolders = Dict{String,String}(savefoldernames .=> mkpath.(joinpath.(settings["dir"], savefoldernames)))
cp(settings_file, joinpath(savefolders["settings"], FILE_PREFIX * "settings.toml"); force = true) # TODO getnow()
clearsavefolders(folders = savefolders) = for (k,f) in folders; rm.(joinpath.(f, readdir(f))); end

# Load and prepare signal data
#   Data:   length H 1D vectors organized into B batches as [H x 1 x B] arrays
#   Labels: length Nout 1D vectors organized into B batches as [Nout x B] arrays
@info "Preparing data..."
make_minibatch(x, y, idxs) = (x[:,:,idxs], y[:,idxs])
const data_set = prepare_data(settings, model_settings)
const train_batches = partition(1:batchsize(data_set[:training_data]), settings["data"]["batch_size"])
const train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches]
# const train_set = [([make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches][1]...,)] # For overtraining testing
const test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:settings["data"]["test_size"])

@info "Plotting random data samples..."
plot_random_data_samples = () -> begin
    plot_xdata =
        settings["data"]["preprocess"]["PCA"]["apply"] ?
            collect(1:heightsize(test_set[1])) :
        settings["data"]["preprocess"]["ilaplace"]["apply"] ?
            log10range(settings["data"]["preprocess"]["ilaplace"]["T2Range"]...; length = settings["data"]["preprocess"]["ilaplace"]["nT2"]) :
        settings["data"]["preprocess"]["wavelet"]["apply"] ?
            collect(1:heightsize(test_set[1])-4) :
        error("No method supplied")
    plot_ydata =
        settings["data"]["preprocess"]["wavelet"]["apply"] ?
            reshape(test_set[1][end - settings["data"]["preprocess"]["wavelet"]["nterms"] + 1 : end,1,:], :, batchsize(test_set[1])) :
        reshape(test_set[1], :, batchsize(test_set[1])) # default
    fig = plot([
        plot(plot_xdata, plot_ydata[:,i];
            xscale = settings["data"]["preprocess"]["ilaplace"]["apply"] ? :log10 : :identity,
            titlefontsize = 8, grid = true, minorgrid = true,
            label = "Data Distbn.",
            title = DrWatson.savename("", data_set[:testing_data_dicts][i][:sweepparams]; connector = ", ")
        ) for i in sample(1:batchsize(plot_ydata), 5; replace = false)
        ]...; layout = (5,1))
    display(fig)
    fig
end
savefig(plot_random_data_samples(), "plots/" * FILE_PREFIX * "datasamples.png")

# Construct model
@info "Constructing model..."
# model = MWFLearning.get_model(settings, model_settings)
# model_summary(model, joinpath(savefolders["models"], FILE_PREFIX * "architecture.txt"))

model = Flux.Chain(
    Sumout(
        @λ(x -> x[1:4,1,:]),
        Flux.Chain(
            @λ(x -> x[5:20,1,:]),
            Flux.Dense(16, 8, Flux.relu),
            ChannelResize(2),
            (ResidualBlock(4, 2, :pre, false) for _ in 1:1)...,
            DenseResize(),
            Flux.Dense(8, 4),
        ),
    ),
    @λ(x -> vcat(Flux.softmax(x[1:2,:]), x[3:4,:])),
)
model_summary(model)

# Compute parameter density, defined as the number of Flux.params / number of training label datapoints
const param_density = sum(length, Flux.params(model)) / sum(b -> length(b[2]), train_set)
@info "Parameter density = $(round(100 * param_density; digits = 2)) %"

# Loss and accuracy function
unitsum(x) = x ./ sum(x)
get_label_weights()::VT = VT(inv.(model_settings["scale"]) .* unitsum(settings["data"]["weights"]))

l2 = @λ (x,y) -> sum((get_label_weights() .* (model(x) .- y)).^2)
mse = @λ (x,y) -> l2(x,y) * 1 // length(y)
crossent = @λ (x,y) -> Flux.crossentropy(model(x), y)
mincrossent = @λ (y) -> -sum(y .* log.(y))

if model_settings["loss"] ∉ ["l2", "mse", "crossent"]
    @warn "Unknown loss $(model_settings["loss"]); defaulting to mse"
    model_settings["loss"] = "mse"
end

loss =
    model_settings["loss"] == "l2" ? l2 :
    model_settings["loss"] == "crossent" ? crossent :
    mse # default

accuracy =
    model_settings["loss"] == "l2" ? @λ( (x,y) -> 100 - 100 * sqrt(loss(x,y) * 1 // length(y)) ) :
    model_settings["loss"] == "crossent" ? @λ( (x,y) -> 100 - 100 * (loss(x,y) - mincrossent(y)) ) :
    @λ( (x,y) -> 100 - 100 * sqrt(loss(x,y)) ) # default

labelerror =
    @λ (x,y) -> 100 .* mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2)

# Optimizer
# opt = Flux.ADAM(
#     settings["optimizer"]["ADAM"]["lr"],
#     (settings["optimizer"]["ADAM"]["beta"]...,))
opt = Flux.ADAM(1e-2, (0.9, 0.999)) #TODO
# opt = Flux.Nesterov(1e-3)

# Callbacks
errs = Dict(
    :training => Dict(:loss => [], :acc => [], :labelerr => []),
    :testing => Dict(:loss => [], :acc => [], :labelerr => []))

train_err_cb = () -> begin
    push!(errs[:training][:loss], mean(Flux.data(loss(b...)) for b in train_set))
    push!(errs[:training][:acc], mean(Flux.data(accuracy(b...)) for b in train_set))
    push!(errs[:training][:labelerr], mean(Flux.data(labelerror(b...)) for b in train_set))
end

test_err_cb = () -> begin
    push!(errs[:testing][:loss], Flux.data(loss(test_set...)))
    push!(errs[:testing][:acc], Flux.data(accuracy(test_set...)))
    push!(errs[:testing][:labelerr], Flux.data(labelerror(test_set...)))
end

plot_errs_cb = () -> begin
    @info " -> Plotting progress..."
    _plot_errs_cb = (k,v) -> begin
        @unpack loss, acc, labelerr = v
        labelerr = permutedims(reduce(hcat, labelerr))
        labelnames = permutedims(settings["data"]["labels"]) # .* " (" .* settings["plot"]["units"] .* ")"
        plot(
            plot(loss;     title = "Loss ($k: min = $(round(minimum(loss); sigdigits = 4)))",      titlefontsize = 10, label = "loss",     legend = :topright, ylim = (minimum(loss), min(1, quantile(loss, 0.90)))),
            plot(acc;      title = "Accuracy ($k: peak = $(round(maximum(acc); sigdigits = 4))%)", titlefontsize = 10, label = "acc",      legend = :topleft,  ylim = (95, 100)),
            plot(labelerr; title = "Label Error ($k: rel. %)",                                     titlefontsize = 10, label = labelnames, legend = :topleft, ylim = (max(0, minimum(labelerr) - 0.5), min(15, quantile(labelerr[:], 0.90)))),
            layout = (1,3)
        )
    end
    fig = plot((_plot_errs_cb(k,v) for (k,v) in errs)...; layout = (length(errs), 1))
    display(fig)
    savefig(fig, "plots/" * FILE_PREFIX * "errs.png")
end

checkpoint_model_opt_cb = () -> begin
    save_time = savebson("models/" * FILE_PREFIX * "model-checkpoint.bson", @dict(model, opt)) #TODO getnow()
    @info " -> Model checkpoint... ($(round(1000*save_time; digits = 2)) ms)"
end

checkpoint_errs_cb = () -> begin
    save_time = savebson("log/" * FILE_PREFIX * "errors.bson", @dict(errs)) #TODO getnow()
    @info " -> Error checkpoint ($(round(1000*save_time; digits = 2)) ms)" #TODO
end

test_err_cb() # initial loss
train_err_cb() # initial loss

cbs = Flux.Optimise.runall([
    Flux.throttle(test_err_cb, 5),
    Flux.throttle(train_err_cb, 5),
    Flux.throttle(plot_errs_cb, 15),
    Flux.throttle(checkpoint_errs_cb, 30),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
])

# Training Loop
const ACC_THRESH = 100.0
const DROP_ETA_THRESH = 1000 # typemax(Int)
const CONVERGED_THRESH = 2500 # typemax(Int)
BEST_ACC = 0.0
LAST_IMPROVED_EPOCH = 0

@info("Beginning training loop...")

try
    for epoch in 1:settings["optimizer"]["epochs"]
        global BEST_ACC, LAST_IMPROVED_EPOCH

        # Train for a single epoch
        Flux.train!(loss, Flux.params(model), train_set, opt; cb = cbs)

        # Calculate accuracy:
        acc = accuracy(test_set...)
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch, acc))
        
        # If our accuracy is good enough, quit out.
        if acc >= ACC_THRESH
            @info " -> Early-exiting: We reached our target accuracy of 99.99%"
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= BEST_ACC
            BEST_ACC = Flux.data(acc)
            LAST_IMPROVED_EPOCH = epoch

            curr_epoch = epoch
            curr_acc = BEST_ACC

            try
                # TODO
                # model = Flux.mapleaves(Flux.data, model) # local scope, can rename
                # save_time = savebson("models/" * FILE_PREFIX * "model.bson", @dict(model, opt, curr_epoch, curr_acc)) #TODO getnow()
                # @info " -> New best accuracy; model saved ($(round(1000*save_time; digits = 2)) ms)"
            catch e
                @warn "Error saving model"
                @warn sprint(showerror, e, catch_backtrace())
            end

            try
                weights = Flux.data.(Flux.params(model))
                save_time = savebson("weights/" * FILE_PREFIX * "weights.bson", @dict(weights, curr_epoch, curr_acc)) #TODO getnow()
                @info " -> New best accuracy; weights saved ($(round(1000*save_time; digits = 2)) ms)"
            catch e
                @warn "Error saving weights"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end

        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch - LAST_IMPROVED_EPOCH >= DROP_ETA_THRESH && opt.eta > 5e-7
            opt.eta /= 2.0
            @warn(" -> Haven't improved in $DROP_ETA_THRESH iters; dropping learning rate to $(opt.eta)")

            # After dropping learning rate, give it a few epochs to improve
            LAST_IMPROVED_EPOCH = epoch
        end

        if epoch - LAST_IMPROVED_EPOCH >= CONVERGED_THRESH
            @warn(" -> Haven't improved in $CONVERGED_THRESH iters; model has converged")
            break
        end
    end
catch e
    if e isa InterruptException
        @warn "Training interrupted by user; breaking out of loop..."
    else
        @warn "Error during training..."
        @warn sprint(showerror, e, catch_backtrace())
    end
end

@info "Plotting prediction histograms..."
model_labels = Flux.data(model(test_set[1]))
true_labels = copy(test_set[2])
fig = plot([
    begin
        scale = settings["plot"]["scale"][i]
        units = settings["plot"]["units"][i]
        err = scale .* (model_labels[i,:] .- true_labels[i,:])
        histogram(err;
            grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["labels"][i] * " ($units)",
            title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
    end for i in 1:size(model_labels, 1)
    ]...)
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "labelhistograms.png")

nothing