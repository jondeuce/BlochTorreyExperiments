# Initialize project/code loading
import Pkg
using Printf
using Statistics: mean, median
using Base.Iterators: repeated, partition
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))

using MWFLearning
using StatsPlots
pyplot(size=(800,600))

# Utils
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
gitdir() = realpath(joinpath(DrWatson.projectdir(), "..")) * "/"

# Settings
settings_file = "settings.toml"
settings = TOML.parsefile(settings_file)
FILE_PREFIX = DrWatson.savename(settings["model"]) * "."

mkpath.(joinpath.(settings["dir"], ("settings", "models", "weights", "log", "plots")))
cp(settings_file, "settings/" * FILE_PREFIX * "settings.toml"; force = true)

# Load and prepare signal data
#   Data:   length H 1D vectors organized into B batches as [H x 1 x B] arrays
#   Labels: length Nout 1D vectors organized into B batches as [Nout x B] arrays
@info "Preparing data..."
make_minibatch(x, y, idxs) = (x[:,:,idxs], y[:,idxs])
data_set = prepare_data(settings)
train_batches = partition(1:batchsize(data_set[:training_data]), settings["data"]["batch_size"])
train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches]
test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:settings["data"]["test_size"])

# Define our model. This is the example model from the Keras documentation,
# "Sequence classification with 1D convolutions", at the following url:
#   https://keras.io/getting-started/sequential-model-guide/

@info "Constructing model..."
model = MWFLearning.get_model(settings)

# Compute parameter density, defined as the number of Flux.params / number of training label datapoints
param_density = sum(length, Flux.params(model)) / sum(b -> length(b[2]), train_set)
@info "Parameter density = $(round(100 * param_density; digits = 2)) %"

# Loss and accuracy function
get_weights()::Union{Vector{Float32},Vector{Float64}} = convert(
    settings["prec"] == 32 ? Vector{Float32} : Vector{Float64},
    settings["data"]["scale"] .* settings["data"]["weights"])

function loss(x,y)
    settings["model"]["loss"] == "l2" ? sum(abs2, get_weights() .* (model(x) .- y)) :
    settings["model"]["loss"] == "crossent" ? Flux.crossentropy(model(x), y) :
    error("Unknown loss function: " * settings["model"]["loss"])
end

mwfloss(x,y) = sum(abs2, model(x)[1,:] .- y[1,:])
accuracy(x,y) = 100 - 100 * sqrt(mwfloss(x,y) / batchsize(y))

# Optimizer
opt = Flux.ADAM(
    settings["optimizer"]["ADAM"]["lr"],
    (settings["optimizer"]["ADAM"]["beta"]...,))

# Callbacks
errs = Dict(
    :training => Dict(:loss => [], :acc => []),
    :testing => Dict(:loss => [], :acc => []))

train_err_cb = () -> begin
    push!(errs[:training][:loss], sum(Flux.data(loss(b...)) for b in train_set))
    push!(errs[:training][:acc], mean(Flux.data(accuracy(b...)) for b in train_set))
end

test_err_cb = () -> begin
    push!(errs[:testing][:loss], Flux.data(loss(test_set...)))
    push!(errs[:testing][:acc], Flux.data(accuracy(test_set...)))
end

plot_errs_cb = () -> begin
    @info " -> Plotting progress..."
    train_loss, train_acc, test_loss, test_acc = errs[:training][:loss], errs[:training][:acc], errs[:testing][:loss], errs[:testing][:acc]
    fig = plot(
        plot(train_loss; title = "Training loss (Min = $(minimum(train_loss)))", label = "loss", ylim = extrema(train_loss[min(5,length(train_loss)):end])),
        plot(train_acc; title = "Training acc (Peak = $(maximum(train_acc))%)", label = "acc", ylim = (95, 100), legend = :bottomright),
        plot(test_loss; title = "Testing loss (Min = $(minimum(test_loss)))", label = "loss", ylim = extrema(test_loss[min(5,length(test_loss)):end])),
        plot(test_acc; title = "Testing acc (Peak = $(maximum(test_acc))%)", label = "acc", ylim = (95, 100), legend = :bottomright);
        layout = (2,2))
    display(fig)
    savefig(fig, "plots/" * FILE_PREFIX * "errs.png")
end

checkpoint_model_opt_cb = () -> begin
    @info " -> Model checkpoint..."
    save_time = @elapsed BSON.bson(
        "models/" * FILE_PREFIX * "model-checkpoint.bson", #TODO getnow()
        @dict(model, opt))
    # save_time = @elapsed DrWatson.@tagsave(
    #     "models/" * "model-checkpoint.bson", #TODO getnow()
    #     @dict(model, curr_epoch, curr_acc),
    #     false, gitdir())
    @info " -> Model checkpoint... (done)"
    # @info " -> Model checkpoint ($save_time s)" #TODO
end

checkpoint_errs_cb = () -> begin
    save_time = @elapsed BSON.bson(
        "log/" * FILE_PREFIX * "errors.bson", #TODO getnow()
        @dict(errs))
    # save_time = @elapsed DrWatson.@tagsave(
    #     "log/" * "$(getnow()).errors.bson",
    #     @dict(errs),
    #     false, gitdir())
    @info " -> Error checkpoint ($save_time s)" #TODO
end

test_err_cb() # initial loss
train_err_cb() # initial loss

cbs = Flux.Optimise.runall([
    Flux.throttle(test_err_cb, 1),
    Flux.throttle(train_err_cb, 3),
    Flux.throttle(plot_errs_cb, 3),
    Flux.throttle(checkpoint_errs_cb, 60),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
    ])

# Training Loop
BEST_ACC = 0.0
LAST_IMPROVED_EPOCH = 0
DROP_ETA_THRESH = typemax(Int)
CONVERGED_THRESH = typemax(Int)

@info("Beginning training loop...")

for epoch in 1:settings["optimizer"]["epochs"]
    global BEST_ACC, LAST_IMPROVED_EPOCH

    # Train for a single epoch
    Flux.train!(loss, Flux.params(model), train_set, opt; cb = cbs)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 99.99
        @info " -> Early-exiting: We reached our target accuracy of 99.99%"
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= BEST_ACC
        BEST_ACC = Flux.data(acc)
        LAST_IMPROVED_EPOCH = epoch

        weights = Flux.data.(Flux.params(model))
        curr_epoch = epoch
        curr_acc = BEST_ACC

        # TODO
        # try
        #     save_time = @elapsed BSON.bson(
        #         "models/" * FILE_PREFIX * "model.bson", #TODO getnow()
        #         @dict(model, opt, curr_epoch, curr_acc))
        #     # save_time = @elapsed DrWatson.@tagsave(
        #     #     "models/" * "model.bson", #TODO getnow()
        #     #     @dict(model, curr_epoch, curr_acc),
        #     #     false, gitdir())
        #     @info " -> New best accuracy; model saved ($save_time s)"
        # catch e
        #     @warn "Error saving model"
        #     @warn sprint(showerror, e, catch_backtrace())
        # end

        # TODO
        try
            save_time = @elapsed BSON.bson(
                "weights/" * FILE_PREFIX * "weights.bson", #TODO getnow()
                @dict(weights, curr_epoch, curr_acc))
            # save_time = @elapsed DrWatson.@tagsave(
            #     "weights/" * "weights.bson", #TODO getnow()
            #     @dict(weights, curr_epoch, curr_acc),
            #     false, gitdir())
            @info " -> New best accuracy; weights saved ($save_time s)"
        catch e
            @warn "Error saving weights"
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch - LAST_IMPROVED_EPOCH >= DROP_ETA_THRESH && opt.eta > 5e-7
        opt.eta /= 2.0
        @warn(" -> Haven't improved in 25 iters; dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        LAST_IMPROVED_EPOCH = epoch
    end

    if epoch - LAST_IMPROVED_EPOCH >= CONVERGED_THRESH
        @warn(" -> Haven't improved in 100 iters; model has converged")
        break
    end
end

"""
NOTES:

Running multiple models for fixed 100 epochs w/ default ADAM

-> Keras_1D_Seq_Class (density = 44%, activation = relu)

                    Softmax
                    Yes                     No
    Dropout
    Yes             98.01% Train Acc        98.66% Train Acc
                    98.01% Test Acc         98.63% Test Acc
                    -> No overtraining      -> No overtraining
    
    No              99.81% Train Acc        99.74% Train Acc
                    99.82% Test Acc         99.76% Test Acc
                    -> No overtraining      -> Yes overtraining

-> Keras_1D_Seq_Class (density = 44%, activation = leakyrelu)

                    Softmax
                    Yes                     No
    Dropout
    No              99.68% Train Acc        99.73% Train Acc
                    99.70% Test Acc         99.73% Test Acc
                    -> No overtraining      -> No overtraining

-> Keras_1D_Seq_Class (density = 17%, activation = leakyrelu, no Dropout)

                    Softmax
                    Yes                     No
    Loss
    MSE             99.64% Train Acc        99.60% Train Acc
                    99.67% Test Acc         99.64% Test Acc
                    -> No overtraining      -> No overtraining

    Crossentropy    99.64% Train Acc        XX.XX% Train Acc
                    99.69% Test Acc         XX.XX% Test Acc
                    -> XX overtraining      -> XX overtraining

"""

nothing