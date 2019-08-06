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
# using CuArrays
using StatsPlots
pyplot(size=(800,450))

# Utils
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
gitdir() = realpath(DrWatson.projectdir(".."))
gitdir() = realpath(DrWatson.projectdir(".."))
gitdir() = realpath(DrWatson.projectdir(".."))
savebson(filename, data::Dict) = @elapsed BSON.bson(filename, data)

# Settings
const settings_file = "settings.toml"
const settings = verify_settings(TOML.parsefile(settings_file))

const DATE_PREFIX = getnow() * "."
const FILE_PREFIX = DATE_PREFIX * DrWatson.savename(settings["model"]) * "."
const GPU = settings["gpu"] :: Bool
const T   = settings["prec"] == 64 ? Float64 : Float32
const VT  = Vector{T}
const MT  = Matrix{T}
const VMT = VecOrMat{T}
const CVT = GPU ? CuVector{T} : Vector{T}

const savefoldernames = ["settings", "models", "weights", "log", "plots"]
const savefolders = Dict{String,String}(savefoldernames .=> mkpath.(joinpath.(settings["dir"], savefoldernames)))
cp(settings_file, joinpath(savefolders["settings"], FILE_PREFIX * "settings.toml"); force = true)
clearsavefolders(folders = savefolders) = for (k,f) in folders; rm.(joinpath.(f, readdir(f))); end

# Load and prepare signal data
@info "Preparing data..."
make_minibatch(x, y, idxs) = (x[.., idxs], y[.., idxs])
const data_set = prepare_data(settings)
GPU && (for k in (:training_data, :testing_data, :training_labels, :testing_labels); data_set[k] = Flux.gpu(data_set[k]); end)

const train_batches = partition(1:batchsize(data_set[:training_data]), settings["data"]["batch_size"]);
const train_set = [make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches];
# const train_set = [([make_minibatch(data_set[:training_data], data_set[:training_labels], i) for i in train_batches][1]...,)]; # For overtraining testing (set batch size small, too)
const test_set = make_minibatch(data_set[:testing_data], data_set[:testing_labels], 1:settings["data"]["test_size"]);

# Construct model
@info "Constructing model..."
model = MWFLearning.get_model(settings);
model = GPU ? Flux.gpu(model) : model;
model_summary(model, joinpath(savefolders["models"], FILE_PREFIX * "architecture.txt"));

#=
Plot example data
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
            test_set[1][end - settings["data"]["preprocess"]["wavelet"]["nterms"] + 1 : end, ..] :
            test_set[1] # default
    fig = plot([
        plot(plot_xdata, Flux.cpu(plot_ydata[:,1,:,i]);
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
=#

# Compute parameter density, defined as the number of Flux.params / number of training label datapoints
test_dofs = length(test_set[2])
train_dofs = sum(batch -> length(batch[2]), train_set)
param_dofs = sum(length, Flux.params(model))
test_param_density = param_dofs / test_dofs
train_param_density = param_dofs / train_dofs
@info @sprintf(" Testing parameter density: %d/%d (%.2f %%)", param_dofs, test_dofs, 100 * test_param_density)
@info @sprintf("Training parameter density: %d/%d (%.2f %%)", param_dofs, train_dofs, 100 * train_param_density)

# Loss and accuracy function
const labelweights = inv.(settings["model"]["scale"]) .* unitsum(settings["data"]["weights"]) |> VT
LabelWeights()::CVT = GPU ? Flux.gpu(copy(labelweights)) : copy(labelweights) |> CVT

l1 = @λ (x,y) -> sum(abs, LabelWeights()::CVT .* (model(x) .- y))
l2 = @λ (x,y) -> sum(abs2, LabelWeights()::CVT .* (model(x) .- y))
mae = @λ (x,y) -> l1(x,y) * 1 // length(y)
mse = @λ (x,y) -> l2(x,y) * 1 // length(y)
crossent = @λ (x,y) -> Flux.crossentropy(model(x), y)
mincrossent = @λ (y) -> -sum(y .* log.(y))

if settings["model"]["loss"] ∉ ["l1", "l2", "mae", "mse", "crossent"]
    @warn "Unknown loss $(settings["model"]["loss"]); defaulting to mse"
    settings["model"]["loss"] = "mse"
end

loss =
    settings["model"]["loss"] == "l1" ? l1 : settings["model"]["loss"] == "mae" ? mae :
    settings["model"]["loss"] == "l2" ? l2 : settings["model"]["loss"] == "mse" ? mse :
    settings["model"]["loss"] == "crossent" ? crossent :
    mse # default

accuracy =
    settings["model"]["acc"] == "mae" ? @λ( (x,y) -> 100 - 100 * mae(x,y) ) :
    settings["model"]["acc"] == "rmse" ? @λ( (x,y) -> 100 - 100 * sqrt(mse(x,y)) ) :
    settings["model"]["acc"] == "crossent" ? @λ( (x,y) -> 100 - 100 * (crossent(x,y) - mincrossent(y)) ) :
    @λ (x,y) -> 100 - 100 * sqrt(mse(x,y)) # default

labelerror =
    # @λ (x,y) -> 100 .* vec(mean(abs.((model(x) .- y) ./ y); dims = 2))
    # @λ (x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2))
    @λ (x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ (maximum(abs.(y); dims = 2) .- minimum(abs.(y); dims = 2)))

# Optimizer
lr(opt) = opt.eta
lr!(opt, α) = (opt.eta = α; opt.eta)
lr(opt::Flux.Optimiser) = lr(opt[1])
lr!(opt::Flux.Optimiser, α) = lr!(opt[1], α)

opt = Flux.ADAM(settings["optimizer"]["ADAM"]["lr"], (settings["optimizer"]["ADAM"]["beta"]...,))
# opt = Flux.Momentum(settings["optimizer"]["SGD"]["lr"], settings["optimizer"]["SGD"]["rho"])
# opt = Flux.ADAMW(settings["optimizer"]["ADAM"]["lr"], (settings["optimizer"]["ADAM"]["beta"]...,), settings["optimizer"]["ADAM"]["decay"])
# opt = MomentumW(settings["optimizer"]["SGD"]["lr"], settings["optimizer"]["SGD"]["rho"], settings["optimizer"]["SGD"]["decay"])

# opt = Flux.Nesterov(1e-1)
# opt = Flux.ADAM(3e-4, (0.9, 0.999))
# opt = Flux.ADAMW(1e-2, (0.9, 0.999), 1e-5)
# opt = MWFLearning.AdaBound(1e-3, (0.9, 0.999), 1e-5, 1e-3)
# opt = Flux.Momentum(1e-4, 0.9)
# opt = Flux.Optimiser(Flux.Momentum(0.1, 0.9), Flux.WeightDecay(1e-4))

# # Fixed learning rate
# LRfun(e) = lr(opt)

# Drop learning rate every LRDROPRATE epochs
LRDROPRATE, LRDROPFACTOR = 50, 10^(1/4)
LRfun(e) = mod(e, LRDROPRATE) == 0 ? lr(opt) / LRDROPFACTOR : lr(opt)

# # Learning rate finder
# LRfun(e) = e <= settings["optimizer"]["epochs"] ?
#     logspace(1,settings["optimizer"]["epochs"],1e-6,0.5)(e) : 0.5

# # Learning rate cycling
# LRSTART, LRMAX, LRMIN = 1e-5, 1e-2, 1e-6
# LRTAIL = settings["optimizer"]["epochs"] ÷ 20
# LRWIDTH = (settings["optimizer"]["epochs"] - LRTAIL) ÷ 2
# LRfun(e) =
#                      e <=   LRWIDTH          ? linspace(        1,            LRWIDTH, LRSTART,   LRMAX)(e) :
#       LRWIDTH + 1 <= e <= 2*LRWIDTH          ? linspace(  LRWIDTH,          2*LRWIDTH,   LRMAX, LRSTART)(e) :
#     2*LRWIDTH + 1 <= e <= 2*LRWIDTH + LRTAIL ? linspace(2*LRWIDTH, 2*LRWIDTH + LRTAIL, LRSTART,   LRMIN)(e) :
#             LRMIN

# Callbacks
CB_EPOCH = 0 # global callback epoch count
CB_EPOCH_RATE = 25 # rate of per epoch callback updates
CB_EPOCH_CHECK(last_epoch) = CB_EPOCH >= last_epoch + CB_EPOCH_RATE
loop_errs = Dict(
    :testing => Dict(:epoch => Int[], :acc => T[]))
errs = Dict(
    :training => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]),
    :testing => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]))
function err_subplots(k,v)
    @unpack epoch, loss, acc, labelerr = v
    labelerr = permutedims(reduce(hcat, labelerr))
    labelnames = permutedims(settings["data"]["labels"]) # .* " (" .* settings["plot"]["units"] .* ")"
    p1 = plot(epoch, loss;     title = "Loss ($k: min = $(round(minimum(loss); sigdigits = 4)))",      lw = 3, titlefontsize = 10, label = "loss",     legend = :topright, ylim = (minimum(loss), quantile(loss, 0.90)))
    p2 = plot(epoch, acc;      title = "Accuracy ($k: peak = $(round(maximum(acc); sigdigits = 4))%)", lw = 3, titlefontsize = 10, label = "acc",      legend = :topleft,  ylim = (90, 100))
    p3 = plot(epoch, labelerr; title = "Label Error ($k: rel. %)",                                     lw = 3, titlefontsize = 10, label = labelnames, legend = :topleft,  ylim = (max(0, minimum(labelerr) - 0.5), min(50, quantile(vec(labelerr), 0.90))))
    (k == :testing) && plot!(p2, loop_errs[:testing][:epoch] .+ 1, loop_errs[:testing][:acc]; label = "loop acc", lw = 2) # Epochs shifted by 1 since accuracy is evaluated after a training within an epoch, whereas callbacks above are called before training
    plot(p1, p2, p3; layout = (1,3))
end
train_err_cb = let LAST_EPOCH = 0
    function()
        CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
        update_time = @elapsed begin
            currloss, curracc, currlaberr = mean([Flux.cpu(Flux.data(loss(b...))) for b in train_set]), mean([Flux.cpu(Flux.data(accuracy(b...))) for b in train_set]), mean([Flux.cpu(Flux.data(labelerror(b...))) for b in train_set])
            push!(errs[:training][:epoch], LAST_EPOCH)
            push!(errs[:training][:loss], currloss)
            push!(errs[:training][:acc], curracc)
            push!(errs[:training][:labelerr], currlaberr)
        end
        @info @sprintf("[%d] -> Updating training error... (%d ms)", LAST_EPOCH, 1000 * update_time)
    end
end
test_err_cb = let LAST_EPOCH = 0
    function()
        CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
        update_time = @elapsed begin
            currloss, curracc, currlaberr = Flux.cpu(Flux.data(loss(test_set...))), Flux.cpu(Flux.data(accuracy(test_set...))), Flux.cpu(Flux.data(labelerror(test_set...)))
            push!(errs[:testing][:epoch], LAST_EPOCH)
            push!(errs[:testing][:loss], currloss)
            push!(errs[:testing][:acc], curracc)
            push!(errs[:testing][:labelerr], currlaberr)
        end
        @info @sprintf("[%d] -> Updating testing error... (%d ms)", LAST_EPOCH, 1000 * update_time)
    end
end
plot_errs_cb = let LAST_EPOCH = 0
    function()
        try
            CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
            plot_time = @elapsed begin
                fig = plot([err_subplots(k,v) for (k,v) in errs]...; layout = (length(errs), 1))
                savefig(fig, "plots/" * FILE_PREFIX * "errs.png")
                display(fig)
            end
            @info @sprintf("[%d] -> Plotting progress... (%d ms)", LAST_EPOCH, 1000 * plot_time)
        catch e
            @info @sprintf("[%d] -> PLOTTING FAILED...", LAST_EPOCH)
        end
    end
end
checkpoint_model_opt_cb = function()
    save_time = @elapsed let opt = MWFLearning.opt_to_cpu(opt, Flux.params(model)), model = Flux.cpu(model)
        savebson("models/" * FILE_PREFIX * "model-checkpoint.bson", @dict(model, opt))
    end
    @info @sprintf("[%d] -> Model checkpoint... (%d ms)", CB_EPOCH, 1000 * save_time)
end
checkpoint_errs_cb = function()
    save_time = @elapsed let errs = deepcopy(errs)
        savebson("log/" * FILE_PREFIX * "errors.bson", @dict(errs))
    end
    @info @sprintf("[%d] -> Error checkpoint... (%d ms)", CB_EPOCH, 1000 * save_time)
end

cbs = Flux.Optimise.runall([
    test_err_cb,
    train_err_cb,
    plot_errs_cb,
    Flux.throttle(checkpoint_errs_cb, 60),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
])

# Training Loop
const ACC_THRESH = 100.0 # Never stop
const DROP_ETA_THRESH = typemax(Int) # Never stop
const CONVERGED_THRESH = typemax(Int) # Never stop
BEST_ACC = 0.0
LAST_IMPROVED_EPOCH = 0

train_loop! = function()
    for epoch in CB_EPOCH .+ (1:settings["optimizer"]["epochs"])
        global BEST_ACC, LAST_IMPROVED_EPOCH, CB_EPOCH
        CB_EPOCH = epoch
        
        # Update learning rate and exit if it has become to small
        last_lr = lr(opt)
        curr_lr = lr!(opt, LRfun(epoch))
        
        (epoch == 1)         &&  @info(" -> Initial learning rate: " * @sprintf("%.2e", curr_lr))
        (last_lr != curr_lr) &&  @info(" -> Learning rate updated: " * @sprintf("%.2e", last_lr) * " --> "  * @sprintf("%.2e", curr_lr))
        (lr(opt) < 1e-6)     && (@info(" -> Early-exiting: Learning rate has dropped below 1e-6"); break)

        # Train for a single epoch
        train_time = @elapsed Flux.train!(loss, Flux.params(model), train_set, opt; cb = cbs) # CuArrays.@sync
        
        # Calculate accuracy:
        acc_time = @elapsed begin
            acc = Flux.cpu(Flux.data(accuracy(test_set...))) # CuArrays.@sync
            push!(loop_errs[:testing][:epoch], epoch)
            push!(loop_errs[:testing][:acc], acc)
        end
        @info @sprintf("[%d] (%d ms): Test accuracy: %.4f (%d ms)", epoch, 1000 * train_time, acc, 1000 * acc_time)

        # If our accuracy is good enough, quit out.
        if acc >= ACC_THRESH
            # @info " -> Early-exiting: We reached our target accuracy of $ACC_THRESH%"
            @info @sprintf("[%d] -> Early-exiting; we reached our target accuracy of %.2f %%", epoch, ACC_THRESH)
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= BEST_ACC
            BEST_ACC = acc
            LAST_IMPROVED_EPOCH = epoch

            # try
            #     # TODO access to undefined reference error?
            #     save_time = @elapsed let opt = MWFLearning.opt_to_cpu(opt, Flux.params(model)), model = Flux.cpu(model)
            #         savebson("models/" * FILE_PREFIX * "model.bson", @dict(model, opt, epoch, acc))
            #     end
            #     # @info " -> New best accuracy; model saved ($(round(1000*save_time; digits = 2)) ms)"
            #     @info @sprintf("[%d] -> New best accuracy; model saved (%d ms)", epoch, 1000 * save_time)
            # catch e
            #     @warn "Error saving model"
            #     @warn sprint(showerror, e, catch_backtrace())
            # end

            try
                save_time = @elapsed let weights = Flux.cpu.(Flux.data.(Flux.params(model)))
                    savebson("weights/" * FILE_PREFIX * "weights.bson", @dict(weights, epoch, acc))
                end
                # @info " -> New best accuracy; weights saved ($(round(1000*save_time; digits = 2)) ms)"
                @info @sprintf("[%d] -> New best accuracy; weights saved (%d ms)", epoch, 1000 * save_time)
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

        # if epoch - LAST_IMPROVED_EPOCH >= CONVERGED_THRESH
        #     @warn(" -> Haven't improved in $CONVERGED_THRESH iters; model has converged")
        #     break
        # end
    end
end

@info("Beginning training loop...")
try
    train_loop!()
catch e
    if e isa InterruptException
        @warn "Training interrupted by user; breaking out of loop..."
    else
        @warn "Error during training..."
        @warn sprint(showerror, e, catch_backtrace())
    end
end

@info "Computing resulting labels..."
model_labels = model(test_set[1]) |> Flux.data |> Flux.cpu |> deepcopy
true_labels = test_set[2] |> Flux.cpu |> deepcopy

#=
@info "Plotting errors vs. learning rate..."
fig = let
    x = [LRfun.(errs[:training][:epoch]), LRfun.(errs[:testing][:epoch])]
    y = [errs[:training][:loss], errs[:testing][:loss]]
    plot(
        plot(x, (e -> log10.(e .- minimum(e) .+ 1e-6)).(y); xscale = :log10, ylabel = "stretched loss ($(settings["model"]["loss"]))", label = ["training" "testing"]),
        plot(x, (e -> log10.(e)).(y); xscale = :log10, xlabel = "learning rate", ylabel = "loss ($(settings["model"]["loss"]))", label = ["training" "testing"]);
        layout = (2,1)
    )
end
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "lossvslearningrate.png")
=#

@info "Plotting prediction histograms..."
prediction_hist = function()
    pred_hist = function(i)
        scale = settings["plot"]["scale"][i]
        units = settings["plot"]["units"][i]
        err = scale .* (model_labels[i,:] .- true_labels[i,:])
        histogram(err;
            grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["labels"][i] * " ($units)",
            title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
    end
    plot([pred_hist(i) for i in 1:size(model_labels, 1)]...)
end
fig = prediction_hist()
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "labelhistograms.png")

@info "Plotting prediction scatter plots..."
prediction_scatter = function()
    pred_scatter = function(i)
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
    plot([pred_scatter(i) for i in 1:size(model_labels, 1)]...)
end
fig = prediction_scatter()
display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "labelscatter.png")

#=
let
    err = model_labels .- true_labels
    mwf_err = err[1,:]
    sp = data_set[:testing_data_dicts][1][:sweepparams]
    datadicts = data_set[:testing_data_dicts]
    xdata = (d -> d[:sweepparams][:T2lp]).(datadicts) - (d -> d[:sweepparams][:T2tiss]).(datadicts)
    p = scatter(xdata, abs.(mwf_err); xlabel = "T2lp - T2tiss", ylabel = "|mwf error|")
end
=#

#=
let
    err = model_labels .- true_labels
    mwf_err = err[1,:]
    sp = data_set[:testing_data_dicts][1][:sweepparams]
    for k in keys(sp)
        xdata = data_set[:testing_data_dicts] .|> d -> d[:sweepparams][k]
        p1 = scatter(xdata, abs.(mwf_err); xlabel = string(k), ylabel = "|mwf error|")
        p2 = scatter(xdata, mwf_err; xlabel = string(k), ylabel = "mwf error")
        p = plot(p1, p2; layout = (1,2), m = (10, :c))
        display(p)
    end
end
=#
