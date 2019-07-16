# Initialize project/code loading
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
include(joinpath(@__DIR__, "../initpaths.jl"))

using Printf
using Statistics: mean, median, std
using StatsBase: quantile, sample, iqr
using Base.Iterators: repeated, partition

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

const DATE_PREFIX = getnow() * "."
const FILE_PREFIX = DATE_PREFIX * DrWatson.savename(model_settings) * "."
const GPU = settings["gpu"] :: Bool
const T   = settings["prec"] == 64 ? Float64 : Float32
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
const data_set = prepare_data(settings)
const train_batches = partition(1:batchsize(data_set[:training_data]), settings["data"]["batch_size"])
const train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches]
# const train_set = [([make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches][1]...,)] # For overtraining testing (set batch size small, too)
const test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:settings["data"]["test_size"])

@info "Plotting random data samples..."
plot_random_data_samples = () -> begin
    plot_xdata =
        settings["data"]["preprocess"]["PCA"]["apply"] ?
            collect(1:heightsize(test_set[1])) :
        settings["data"]["preprocess"]["ilaplace"]["apply"] ?
            log10range(settings["data"]["preprocess"]["ilaplace"]["T2Range"]...; length = settings["data"]["preprocess"]["ilaplace"]["nT2"]) :
        settings["data"]["preprocess"]["wavelet"]["apply"] ?
            collect(1:heightsize(test_set[1])) :
        settings["data"]["preprocess"]["chunk"]["apply"] ?
            collect(1:settings["data"]["preprocess"]["chunk"]["size"]) :
            collect(1:heightsize(test_set[1])) # Default
    plot_ydata =
        settings["data"]["preprocess"]["wavelet"]["apply"] ?
            test_set[1][end - settings["data"]["preprocess"]["wavelet"]["nterms"] + 1 : end, 1:1, :] :
            test_set[1] # default
    fig = plot([
        plot(plot_xdata, plot_ydata[:,:,i];
            xscale = settings["data"]["preprocess"]["ilaplace"]["apply"] ? :log10 : :identity,
            titlefontsize = 8, grid = true, minorgrid = true,
            legend = :none, #label = "Data Distbn.",
            title = DrWatson.savename("", data_set[:testing_data_dicts][i][:sweepparams]; connector = ", ")
        ) for i in sample(1:batchsize(plot_ydata), 5; replace = false)
        ]...; layout = (5,1))
    display(fig)
    fig
end
savefig(plot_random_data_samples(), "plots/" * FILE_PREFIX * "datasamples.png")

# Construct model
@info "Constructing model..."
model = MWFLearning.get_model(settings);
model_summary(model, joinpath(savefolders["models"], FILE_PREFIX * "architecture.txt"));

# Compute parameter density, defined as the number of Flux.params / number of training label datapoints
const param_density = sum(length, Flux.params(model)) / sum(b -> length(b[2]), train_set)
@info "Parameter density = $(round(100 * param_density; digits = 2)) %"

# Loss and accuracy function
unitsum(x) = x ./ sum(x)
LabelWeights()::VT = VT(inv.(model_settings["scale"]) .* unitsum(settings["data"]["weights"]))

l1 = @λ (x,y) -> sum(abs.(LabelWeights() .* (model(x) .- y)))
l2 = @λ (x,y) -> sum((LabelWeights() .* (model(x) .- y)).^2)
mae = @λ (x,y) -> l1(x,y) * 1 // length(y)
mse = @λ (x,y) -> l2(x,y) * 1 // length(y)
crossent = @λ (x,y) -> Flux.crossentropy(model(x), y)
mincrossent = @λ (y) -> -sum(y .* log.(y))

if model_settings["loss"] ∉ ["l1", "l2", "mae", "mse", "crossent"]
    @warn "Unknown loss $(model_settings["loss"]); defaulting to mse"
    model_settings["loss"] = "mse"
end

loss =
    model_settings["loss"] == "l1" ? l1 : model_settings["loss"] == "mae" ? mae :
    model_settings["loss"] == "l2" ? l2 : model_settings["loss"] == "mse" ? mse :
    model_settings["loss"] == "crossent" ? crossent :
    mse # default

accuracy =
    model_settings["acc"] == "mae" ? @λ( (x,y) -> 100 - 100 * mae(x,y) ) :
    model_settings["acc"] == "rmse" ? @λ( (x,y) -> 100 - 100 * sqrt(mse(x,y)) ) :
    model_settings["acc"] == "crossent" ? @λ( (x,y) -> 100 - 100 * (crossent(x,y) - mincrossent(y)) ) :
    @λ (x,y) -> 100 - 100 * sqrt(mse(x,y)) # default

labelerror =
    # @λ (x,y) -> 100 .* vec(mean(abs.((model(x) .- y) ./ y); dims = 2))
    # @λ (x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2))
    @λ (x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ (e->e[2]-e[1]).(extrema(y; dims=2)))

# Utils
linspace(x1,x2,y1,y2) = x -> (y2-y1)/(x2-x1) * (x-x1) + y1
logspace(x1,x2,y1,y2) = x -> 10^linspace(x1,x2,log10(y1),log10(y2))(x)

# Optimizer
lr(opt) = opt.eta
lr!(opt, α) = (opt.eta = α; opt.eta)
lr(opt::Flux.Optimiser) = lr(opt[1])
lr!(opt::Flux.Optimiser, α) = lr!(opt[1], α)

opt = Flux.ADAM(settings["optimizer"]["ADAM"]["lr"], (settings["optimizer"]["ADAM"]["beta"]...,))
# opt = Flux.Nesterov(1e-1)
# opt = Flux.ADAM(1e-2, (0.9, 0.999))
# opt = Flux.ADAM(3e-4, (0.9, 0.999))
# opt = Flux.ADAMW(1e-2, (0.9, 0.999), 1e-5)
# opt = Flux.ADAMW(1e-3, (0.9, 0.999), 1e-5)
# opt = Flux.ADAMW(3e-4, (0.9, 0.999), 1e-5)
# opt = Flux.ADAMW(1e-4, (0.9, 0.999), 1e-5)
# opt = MWFLearning.AdaBound(1e-3, (0.9, 0.999), 1e-5, 1e-3)
# opt = Flux.Momentum(1e-3, 0.9)
# opt = Flux.Momentum(3e-4, 0.9)
# opt = Flux.Momentum(1e-4, 0.9)

# Fixed learning rate
LRfun(e) = lr(opt)

# # Learning rate finder
# LRfun(e) = e <= settings["optimizer"]["epochs"] ?
#     logspace(1,settings["optimizer"]["epochs"],1e-6,0.5)(e) : 0.5

# # Learning rate cycling
# LRSTART, LRMAX, LRMIN = 1e-5, 1e-2, 1e-6
# LRTAIL = settings["optimizer"]["epochs"] ÷ 20
# LRWIDTH = (settings["optimizer"]["epochs"] - LRTAIL) ÷ 2
# LRfun(e) =
#                      e <=   LRWIDTH          ? linspace(        1,            LRWIDTH, LRSTART, LRMAX)(e) :
#       LRWIDTH + 1 <= e <= 2*LRWIDTH          ? linspace(  LRWIDTH,          2*LRWIDTH, LRMAX,   LRSTART)(e) :
#     2*LRWIDTH + 1 <= e <= 2*LRWIDTH + LRTAIL ? linspace(2*LRWIDTH, 2*LRWIDTH + LRTAIL, LRSTART, LRMIN)(e) :
#     LRMIN

# Callbacks
CB_EPOCH = 0 # global callback epoch count
CB_EPOCH_RATE = 25 # rate of per epoch callback updates
CB_EPOCH_CHECK(last_epoch) = CB_EPOCH >= last_epoch + CB_EPOCH_RATE
errs = Dict(
    :training => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]),
    :testing => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]))
train_err_cb = let LAST_EPOCH = 0
    function()
        CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
        update_time = @elapsed begin
            currloss, curracc, currlaberr = mean(Flux.data(loss(b...)) for b in train_set), mean(Flux.data(accuracy(b...)) for b in train_set), mean(Flux.data(labelerror(b...)) for b in train_set)
            push!(errs[:training][:epoch], CB_EPOCH)
            push!(errs[:training][:loss], currloss)
            push!(errs[:training][:acc], curracc)
            push!(errs[:training][:labelerr], currlaberr)
        end
        @info " -> Updating training error... ($(round(1000*update_time; digits = 2)) ms)"
    end
end
test_err_cb = let LAST_EPOCH = 0
    function()
        CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
        update_time = @elapsed begin
            currloss, curracc, currlaberr = Flux.data(loss(test_set...)), Flux.data(accuracy(test_set...)), Flux.data(labelerror(test_set...))
            push!(errs[:testing][:epoch], CB_EPOCH)
            push!(errs[:testing][:loss], currloss)
            push!(errs[:testing][:acc], curracc)
            push!(errs[:testing][:labelerr], currlaberr)
        end
        @info " -> Updating testing error... ($(round(1000*update_time; digits = 2)) ms)"
    end
end
plot_errs_cb = let LAST_EPOCH = 0
    function make_subplot(k,v)
        @unpack epoch, loss, acc, labelerr = v
        labelerr = permutedims(reduce(hcat, labelerr))
        labelnames = permutedims(settings["data"]["labels"]) # .* " (" .* settings["plot"]["units"] .* ")"
        plot(
            plot(epoch, loss;     title = "Loss ($k: min = $(round(minimum(loss); sigdigits = 4)))",      lw = 3, titlefontsize = 10, label = "loss",     legend = :topright, ylim = (minimum(loss), quantile(loss, 0.90))),
            plot(epoch, acc;      title = "Accuracy ($k: peak = $(round(maximum(acc); sigdigits = 4))%)", lw = 3, titlefontsize = 10, label = "acc",      legend = :topleft,  ylim = (90, 100)),
            plot(epoch, labelerr; title = "Label Error ($k: rel. %)",                                     lw = 3, titlefontsize = 10, label = labelnames, legend = :topleft,  ylim = (max(0, minimum(labelerr) - 0.5), min(50, quantile(vec(labelerr), 0.90)))),
            layout = (1,3)
        )
    end
    function()
        try
            CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
            plot_time = @elapsed begin
                fig = plot([make_subplot(k,v) for (k,v) in errs]...; layout = (length(errs), 1))
                savefig(fig, "plots/" * FILE_PREFIX * "errs.png")
                display(fig)
            end
            @info " -> Plotting progress... ($(round(1000*plot_time; digits = 2)) ms)"
        catch e
            @info " -> Plotting FAILED..."
        end
    end
end
checkpoint_model_opt_cb = function()
    save_time = savebson("models/" * FILE_PREFIX * "model-checkpoint.bson", @dict(model, opt)) #TODO getnow()
    @info " -> Model checkpoint... ($(round(1000*save_time; digits = 2)) ms)"
end
checkpoint_errs_cb = function()
    save_time = savebson("log/" * FILE_PREFIX * "errors.bson", @dict(errs)) #TODO getnow()
    @info " -> Error checkpoint ($(round(1000*save_time; digits = 2)) ms)" #TODO
end

test_err_cb() # initial loss
train_err_cb() # initial loss

cbs = Flux.Optimise.runall([
    Flux.throttle(test_err_cb, 3),
    Flux.throttle(train_err_cb, 3),
    Flux.throttle(plot_errs_cb, 60),
    Flux.throttle(checkpoint_errs_cb, 60),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
])

# Training Loop
const ACC_THRESH = 100.0 # Never stop
const DROP_ETA_THRESH = typemax(Int) # 250 TODO
const CONVERGED_THRESH = typemax(Int) # 500 TODO
BEST_ACC = 0.0
LAST_IMPROVED_EPOCH = 0

@info("Beginning training loop...")

try
    for epoch in CB_EPOCH .+ (1:settings["optimizer"]["epochs"]) #1:typemax(Int) #TODO
        global BEST_ACC, LAST_IMPROVED_EPOCH, CB_EPOCH
        CB_EPOCH = epoch
        
        # Set the learning rate
        lr!(opt, LRfun(epoch))

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
                # let model = Flux.mapleaves(Flux.data, model) # local scope, can rename
                #     save_time = savebson("models/" * FILE_PREFIX * "model.bson", @dict(model, opt, curr_epoch, curr_acc)) #TODO getnow()
                #     @info " -> New best accuracy; model saved ($(round(1000*save_time; digits = 2)) ms)"
                # end
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

        # # If we haven't seen improvement in 5 epochs, drop our learning rate:
        # if epoch - LAST_IMPROVED_EPOCH >= DROP_ETA_THRESH && lr(opt) > 1e-6
        #     lr!(opt, lr(opt)/2)
        #     @warn(" -> Haven't improved in $DROP_ETA_THRESH iters; dropping learning rate to $(lr(opt))")
        # 
        #     # After dropping learning rate, give it a few epochs to improve
        #     LAST_IMPROVED_EPOCH = epoch
        # end

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

@info "Computing resulting labels..."
model_labels = Flux.data(model(test_set[1]))
true_labels = copy(test_set[2])

@info "Plotting errors vs. learning rate..."
fig = let
    x = [LRfun.(errs[:training][:epoch]), LRfun.(errs[:testing][:epoch])]
    y = [errs[:training][:loss], errs[:testing][:loss]]
    plot(
        plot(x, (e -> log10.(e .- minimum(e) .+ 1e-6)).(y); xscale = :log10, ylabel = "stretched loss ($(model_settings["loss"]))", label = ["training" "testing"]),
        plot(x, (e -> log10.(e)).(y); xscale = :log10, xlabel = "learning rate", ylabel = "loss ($(model_settings["loss"]))", label = ["training" "testing"]);
        layout = (2,1)
    )
end
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "lossvslearningrate.png")

@info "Plotting prediction histograms..."
prediction_hist = function(i)
    scale = settings["plot"]["scale"][i]
    units = settings["plot"]["units"][i]
    err = scale .* (model_labels[i,:] .- true_labels[i,:])
    histogram(err;
        grid = true, minorgrid = true, titlefontsize = 10,
        label = settings["data"]["labels"][i] * " ($units)",
        title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
    )
end
fig = plot([prediction_hist(i) for i in 1:size(model_labels, 1)]...)
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "labelhistograms.png")

@info "Plotting prediction scatter plots..."
prediction_scatter = function(i)
    scale = settings["plot"]["scale"][i]
    units = settings["plot"]["units"][i]
    datascale = scale * settings["model"]["scale"][i]
    p = scatter(scale * true_labels[i,:], scale * model_labels[i,:];
        marker = :circle, grid = true, minorgrid = true, titlefontsize = 10,
        label = settings["data"]["labels"][i] * " ($units)",
        # title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
    )
    plot!(p, identity, ylims(p)...; line = (:dash, 2, :red), label = L"y = x")
end
fig = plot([prediction_scatter(i) for i in 1:size(model_labels, 1)]...)
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "labelscatter.png")

nothing

# let
#     err = Flux.data(model(test_set[1])) .- test_set[2]
#     mwf_err = err[1,:]
#     for k in keys(data_set[:testing_data_dicts][1][:sweepparams])
#         xdata = data_set[:testing_data_dicts] .|> d -> d[:sweepparams][k]
#         p = scatter(xdata, abs.(mwf_err); m = (10, :c), xlabel = string(k), ylabel = "mwf_err")
#         display(p)
#     end
# end
