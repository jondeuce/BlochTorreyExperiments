# Initialize project packages
import Pkg
using Printf
using Base.Iterators: repeated, partition
Pkg.activate(joinpath(@__DIR__, "../.."))
include(joinpath(@__DIR__, "../../initpaths.jl"))

# Code loading
using MWFLearning
settings = TOML.parsefile(joinpath(@__DIR__, "convnet_settings.toml")) #TODO

# Plotting
using StatsPlots
pyplot(size=(1200,900))

# Utils
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
gitdir() = realpath(joinpath(@__DIR__, "../../..")) * "/"
prnt = Ref(false)
print_size(i) = prnt[] ?
    x -> (println("Step $i: size(x) = $(size(x))"); x) :
    identity

# Load and prepare signal data
@info "Preparing data..."
make_minibatch(x, y, idxs) = (x[:,:,idxs], y[:,:,idxs])
data_set = prepare_data(settings)
train_mb_idxs = partition(1:size(data_set[:training_data], 3), settings["data"]["batch_size"])
train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_mb_idxs]
test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:size(data_set[:testing_data], 3))

# Define our model. This is the example model from the Keras documentation,
# "Sequence classification with 1D convolutions", at the following url:
#   https://keras.io/getting-started/sequential-model-guide/

@info "Constructing model..."
H = settings["data"]["nT2"] # data height
C = 1 # number of channels
Nfeat = [32,64] # features per conv layer
Npool = 4 # max/mean pooling size
Nkern = 15 # kernel size
Npad = Nkern ÷ 2 # pad size
Ndense = Nfeat[end] * ((H ÷ Npool) ÷ Npool)

model = Chain(
    # Print initial data size
    print_size(0),

    # Two convolution layers followed by max pooling
    Conv((Nkern,), C => Nfeat[1], pad = (Npad,), relu), # (H, 1, 1) -> (H, Nfeat[1], 1)
    print_size(1),
    Conv((Nkern,), Nfeat[1] => Nfeat[1], pad = (Npad,), relu), # (H, Nfeat[1], 1) -> (H, Nfeat[1], 1)
    print_size(2),
    MaxPool((Npool,)), # (H, Nfeat[1], 1) -> (H/Npool, Nfeat[1], 1)
    print_size(3),

    # Two more convolution layers followed by mean pooling
    Conv((Nkern,), Nfeat[1] => Nfeat[2], pad = (Npad,), relu), # (H/Npool, Nfeat[1], 1) -> (H/Npool, Nfeat[2], 1)
    print_size(4),
    Conv((Nkern,), Nfeat[2] => Nfeat[2], pad = (Npad,), relu), # (H/Npool, Nfeat[2], 1) -> (H/Npool, Nfeat[2], 1)
    print_size(5),
    MeanPool((Npool,)), # (H/Npool, Nfeat[2], 1) -> (H/Npool^2, Nfeat[2], 1)
    print_size(6),

    # Dropout layer
    Dropout(0.5),
    print_size(7),

    # Dense layer
    x -> reshape(x, :, size(x, 3)),
    print_size(8),
    Dense(Ndense, 1, relu),
    print_size(9),
    x -> reshape(x, :, 1, length(x)),
    print_size(10),
)

# Loss and accuracy function
loss(x,y) = sum(abs2, model(x) .- y)
accuracy(x,y) = 100 - 100 * sqrt(loss(x,y) / length(y))

@info("Beginning training loop...")
# Save paths
mkpath.(joinpath.(settings["dir"], ("models", "weights")))

# Optimizer
opt = ADAM(
    settings["optimizer"]["ADAM"]["lr"],
    (settings["optimizer"]["ADAM"]["beta"]...,))

best_acc = 0.0
last_improvement = 0

for epoch_idx in 1:settings["optimizer"]["epochs"]
    global best_acc, last_improvement

    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 99.99
        @info(" -> Early-exiting: We reached our target accuracy of 99.99%")
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
        @time try
            @info(" -> New best accuracy; saving model")
            BSON.bson(
                "models/" * "model.bson", #TODO getnow()
                @dict(model, curr_epoch, curr_acc))
            # DrWatson.@tagsave(
            #     "models/" * "model.bson", #TODO getnow()
            #     @dict(model, curr_epoch, curr_acc),
            #     false, gitdir())
        catch e
            @warn "Error saving model"
            @warn sprint(showerror, e, catch_backtrace())
        end

        # TODO
        @time try
            @info(" -> New best accuracy; saving weights")
            BSON.bson(
                "weights/" * "weights.bson", #TODO getnow()
                @dict(weights, curr_epoch, curr_acc))
            # DrWatson.@tagsave(
            #     "weights/" * "weights.bson", #TODO getnow()
            #     @dict(weights, curr_epoch, curr_acc),
            #     false, gitdir())
        catch e
            @warn "Error saving weights"
            @warn sprint(showerror, e, catch_backtrace())
        end
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 25 && opt.eta > 1e-6
        opt.eta /= 2.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 50
        @warn(" -> We're calling this converged.")
        break
    end
end