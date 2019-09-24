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
const data_set = prepare_data(settings)
GPU && (for k in (:training_data, :testing_data, :training_thetas, :testing_thetas); data_set[k] = Flux.gpu(data_set[k]); end)

train_set, test_set = if settings["model"]["problem"] == "forward"
    flatten = x -> reshape(x, :, batchsize(x))
    training_batches(data_set[:training_thetas], flatten(data_set[:training_data]), settings["data"]["train_batch"]),
    testing_batches(data_set[:testing_thetas], flatten(data_set[:testing_data]))
else
    training_batches(data_set[:training_data], data_set[:training_thetas], settings["data"]["train_batch"]),
    testing_batches(data_set[:testing_data], data_set[:testing_thetas])
end

# Labels (outputs) and features (inputs)
features, labels = @λ(batch -> batch[1]), @λ(batch -> batch[2])
signals, thetas = @λ(batch -> batch[1]), @λ(batch -> batch[2])
if settings["model"]["problem"] == "forward"; signals, thetas = thetas, signals; end

# Construct model
@info "Constructing model..."
model = MWFLearning.make_models(settings);
model = GPU ? Flux.gpu(model) : model;
model_summary(model, joinpath(savefolders["models"], FILE_PREFIX * "architecture.txt"));
param_summary(model, train_set, test_set)

# # Construct model
# @info "Constructing model..."
# model, discrm, sampler = MWFLearning.make_models(settings);
# model  = GPU ? Flux.gpu(model)  : model;
# discrm = GPU ? Flux.gpu(discrm) : discrm;
# model_summary(model,  joinpath(savefolders["models"], FILE_PREFIX * "generator.architecture.txt"));
# model_summary(discrm, joinpath(savefolders["models"], FILE_PREFIX * "discriminator.architecture.txt"));
# param_summary(model, train_set, test_set);
# param_summary(discrm, train_set, test_set);

# Loss and accuracy function
function getlabelweights()::Union{CVT, Nothing}
    if settings["model"]["problem"] == "forward"
        nothing
    else
        w = inv.(settings["data"]["info"]["labwidth"]) .* unitsum(settings["data"]["info"]["labweights"]) |> copy |> VT
        w = (GPU ? Flux.gpu(w) : w) |> CVT
    end
end
@unpack loss, accuracy, labelacc = makelosses(model, settings["model"]["loss"], getlabelweights())

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

# Fixed learning rate
LRfun(e) = MWFLearning.fixedlr(e,opt)

# Global error dicts
loop_errs = Dict(
    :testing => Dict(:epoch => Int[], :acc => T[]))
errs = Dict(
    :training => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]),
    :testing => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]))

# Callbacks
CB_EPOCH = 0 # global callback epoch count
CB_EPOCH_RATE = 25 # rate of per epoch callback updates
CB_EPOCH_CHECK(last_epoch) = CB_EPOCH >= last_epoch + CB_EPOCH_RATE
function err_subplots(k,v)
    @unpack epoch, loss, acc, labelerr = v
    labelerr = permutedims(reduce(hcat, labelerr))
    labelnames = permutedims(settings["data"]["info"]["labnames"]) # .* " (" .* settings["data"]["info"]["labunits"] .* ")"
    labelcolor = permutedims(RGB[cgrad(:darkrainbow)[z] for z in range(0.0, 1.0, length = size(labelerr,2))])
    labellegend = settings["model"]["problem"] == "forward" ? nothing : :topleft
    p1 = plot(epoch, loss;     title = "Loss ($k: min = $(round(minimum(loss); sigdigits = 4)))",                      lw = 3, titlefontsize = 10, label = "loss",     legend = :topright,   ylim = (minimum(loss), quantile(loss, 0.95)))
    p2 = plot(epoch, acc;      title = "Accuracy ($k: peak = $(round(maximum(acc); sigdigits = 4))%)",                 lw = 3, titlefontsize = 10, label = "acc",      legend = :topleft,    ylim = (clamp(maximum(acc), 50, 99) - 0.5, 100))
    p3 = plot(epoch, labelerr; title = "Label Error ($k: rel. %)",                                     c = labelcolor, lw = 3, titlefontsize = 10, label = labelnames, legend = labellegend, ylim = (max(0, minimum(vec(labelerr)) - 1.0), min(50, 1.2 * maximum(labelerr[end,:])))) #min(50, quantile(vec(labelerr), 0.90))
    (k == :testing) && plot!(p2, loop_errs[:testing][:epoch] .+ 1, loop_errs[:testing][:acc]; label = "loop acc", lw = 2) # Epochs shifted by 1 since accuracy is evaluated after a training within an epoch, whereas callbacks above are called before training
    plot(p1, p2, p3; layout = (1,3))
end
train_err_cb = let LAST_EPOCH = 0
    function()
        CB_EPOCH_CHECK(LAST_EPOCH) ? (LAST_EPOCH = CB_EPOCH) : return nothing
        update_time = @elapsed begin
            currloss, curracc, currlaberr = mean([Flux.cpu(Flux.data(loss(b...))) for b in train_set]), mean([Flux.cpu(Flux.data(accuracy(b...))) for b in train_set]), mean([Flux.cpu(Flux.data(labelacc(b...))) for b in train_set])
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
            currloss, curracc, currlaberr = Flux.cpu(Flux.data(loss(test_set...))), Flux.cpu(Flux.data(accuracy(test_set...))), Flux.cpu(Flux.data(labelacc(test_set...)))
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
true_signals  = signals(test_set)  |> Flux.cpu |> deepcopy
true_thetas   = thetas(test_set)   |> Flux.cpu |> deepcopy
model_signals = features(test_set) |> Flux.data |> Flux.cpu |> deepcopy
model_thetas  = model(features(test_set)) |> Flux.data |> Flux.cpu |> deepcopy
if settings["model"]["problem"] == "forward"; model_signals, model_thetas = model_thetas, model_signals; end

error("got here")

prediction_hist = function()
    pred_hist = function(i)
        scale = settings["data"]["info"]["labwidth"][i]
        units = settings["data"]["info"]["labunits"][i]
        err = scale .* (model_thetas[i,:] .- true_thetas[i,:])
        histogram(err;
            grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["info"]["labnames"][i] * " ($units)",
            title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
    end
    plot([pred_hist(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting prediction histograms..."
fig = prediction_hist()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "labelhistograms.png")

prediction_scatter = function()
    pred_scatter = function(i)
        scale = settings["data"]["info"]["labwidth"][i]
        units = settings["data"]["info"]["labunits"][i]
        datascale = scale * settings["data"]["info"]["labwidth"][i]
        p = scatter(scale * true_thetas[i,:], scale * model_thetas[i,:];
            marker = :circle, grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["info"]["labnames"][i] * " ($units)",
            # title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
        plot!(p, identity, ylims(p)...; line = (:dash, 2, :red), label = L"y = x")
    end
    plot([pred_scatter(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting prediction scatter plots..."
fig = prediction_scatter()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "labelscatter.png")

forward_plot = function()
    forward_rmse = function(i)
        y = sum(signals(test_set)[:,1,:,i]; dims = 2) # Assumes signal is split linearly into channels
        z_class = MWFLearning.myelin_prop(true_thetas[:,i]...) # Forward simulation of true parameters
        z_model =
            settings["model"]["problem"] == "forward" ? model_signals : # Forward simulated signal from trained model
            MWFLearning.myelin_prop(model_thetas[:,i]...) # Forward simulation of model predicted parameters
        return (e_class = rmsd(y, z_class), e_model = rmsd(y, z_model))
    end
    errors = [forward_rmse(i) for i in 1:batchsize(features(test_set))]
    e_class = (e -> e.e_class).(errors)
    e_model = (e -> e.e_model).(errors)
    p = scatter([e_class e_model];
        labels = ["RMSE: Classical" "RMSE: Model"],
        marker = [:circle :square],
        grid = true, minorgrid = true, titlefontsize = 10, ylim = (0, 0.05)
    )
end
@info "Plotting forward simulation error plots..."
fig = forward_plot()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "forwarderror.png")

errorvslr = function()
    x = [LRfun.(errs[:training][:epoch]), LRfun.(errs[:testing][:epoch])]
    y = [errs[:training][:loss], errs[:testing][:loss]]
    plot(
        plot(x, (e -> log10.(e .- minimum(e) .+ 1e-6)).(y); xscale = :log10, ylabel = "stretched loss ($(settings["model"]["loss"]))", label = ["training" "testing"]),
        plot(x, (e -> log10.(e)).(y); xscale = :log10, xlabel = "learning rate", ylabel = "loss ($(settings["model"]["loss"]))", label = ["training" "testing"]);
        layout = (2,1)
    )
end
# @info "Plotting errors vs. learning rate..."
# fig = errorvslr()
# display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "lossvslearningrate.png")

errorvsthetas = function()
    err = model_thetas .- true_thetas
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
# errorvsthetas()