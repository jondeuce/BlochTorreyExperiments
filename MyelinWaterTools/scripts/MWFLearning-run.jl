# Initialize project packages
import Pkg
using Printf
using Base.Iterators: repeated, partition
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "../initpaths.jl"))

# Code loading
using MWFLearning
settings = TOML.parsefile("settings.toml")

# Plotting
using StatsPlots
pyplot(size=(800,600))

# Utils
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
gitdir() = realpath(joinpath(@__DIR__, "../..")) * "/"

# Load and prepare signal data
@info "Preparing data..."
make_minibatch(x, y, idxs) = (x[:,:,idxs], y[:,:,idxs])
data_set = prepare_data(settings)
train_batches = partition(1:size(data_set[:training_data], 3), settings["data"]["batch_size"])
train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches]
test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:settings["data"]["test_size"])

# Define our model. This is the example model from the Keras documentation,
# "Sequence classification with 1D convolutions", at the following url:
#   https://keras.io/getting-started/sequential-model-guide/

@info "Constructing model..."
Nfeat = [4,8] # features per conv layer
Npool = 4 # max/mean pooling size
Nkern = 15 # kernel size
Nout = 2 # number of output labels
model = MWFLearning.get_model(settings, @ntuple(Nfeat, Npool, Nkern, Nout))

# Loss and accuracy function
function loss(x,y)
    ŷ = model(x)
    w = eltype(x)[1, 1e-1]
    return sum(abs2, w .* (ŷ - y))
end
mwfloss(x,y) = sum(abs2, model(x)[1,:,:] .- y[1,:,:])
accuracy(x,y) = 100 - 100 * sqrt(mwfloss(x,y) / size(y,3))

# Optimizer
opt = ADAM(
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
    fig = plot(
        plot(errs[:training][:loss]; title = "Training loss", label = "loss", ylim = (0, 5 * median(errs[:training][:loss]))),
        plot(errs[:training][:acc]; title = "Training acc", label = "acc", ylim = (95, 100), legend = :bottomright),
        plot(errs[:testing][:loss]; title = "Testing loss", label = "loss", ylim = (0, 5 * median(errs[:testing][:loss]))),
        plot(errs[:testing][:acc]; title = "Testing acc", label = "acc", ylim = (95, 100), legend = :bottomright);
        layout = (2,2))
    display(fig)
    savefig(fig, "plots/" * "errs.png")
end

checkpoint_model_opt_cb = () -> begin
    @info " -> Model checkpoint..."
    save_time = @elapsed BSON.bson(
        "models/" * "model-checkpoint.bson", #TODO getnow()
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
        "log/" * "errors.bson", #TODO getnow()
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
    Flux.throttle(train_err_cb, 15),
    Flux.throttle(plot_errs_cb, 15),
    Flux.throttle(checkpoint_errs_cb, 60),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
    ])

# Training Loop
@info "Beginning training loop..."
mkpath.(joinpath.(settings["dir"], ("models", "weights", "log", "plots")))
best_acc = 0.0
last_improvement = 0

for epoch_idx in 1:settings["optimizer"]["epochs"]
    global best_acc, last_improvement

    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt; cb = cbs)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 99.99
        @info " -> Early-exiting: We reached our target accuracy of 99.99%"
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        best_acc = Flux.data(acc)
        last_improvement = epoch_idx

        weights = Flux.data.(params(model))
        curr_epoch = epoch_idx
        curr_acc = best_acc

        # TODO
        # try
        #     save_time = @elapsed BSON.bson(
        #         "models/" * "model.bson", #TODO getnow()
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
                "weights/" * "weights.bson", #TODO getnow()
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
    if epoch_idx - last_improvement >= 25 && opt.eta > 5e-7
        opt.eta /= 2.0
        @warn(" -> Haven't improved in 25 iters; dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 100
        @warn(" -> Haven't improved in 100 iters; model has converged")
        break
    end
end