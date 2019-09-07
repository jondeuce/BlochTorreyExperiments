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
const data_set = prepare_data(settings)
GPU && (for k in (:training_data, :testing_data, :training_thetas, :testing_thetas); data_set[k] = Flux.gpu(data_set[k]); end)

flatsignal = x -> reshape(x, :, batchsize(x))
train_fwd = training_batches(data_set[:training_thetas], flatsignal(data_set[:training_data]), settings["data"]["batch_size"])
train_inv = training_batches(data_set[:training_data], data_set[:training_thetas], settings["data"]["batch_size"])
test_fwd = testing_batches(data_set[:testing_thetas], flatsignal(data_set[:testing_data]))
test_inv = testing_batches(data_set[:testing_data], data_set[:testing_thetas])

# Construct model
@info "Constructing model..."
genatr, discrm, sampler = MWFLearning.get_model(settings);
genatr = GPU ? Flux.gpu(genatr) : genatr;
discrm = GPU ? Flux.gpu(discrm) : discrm;
model_summary(genatr,  joinpath(savefolders["models"], FILE_PREFIX * "generator.architecture.txt"));
model_summary(discrm, joinpath(savefolders["models"], FILE_PREFIX * "discriminator.architecture.txt"));
param_summary(genatr, train_fwd, test_fwd);
param_summary(discrm, train_fwd, test_fwd);

# Loss and accuracy function
function thetaweights()::CVT
    w = inv.(settings["model"]["scale"]) .* unitsum(settings["data"]["weights"]) |> copy |> VT
    w = (GPU ? Flux.gpu(w) : w) |> CVT
    return w::CVT
end
fwdloss, fwdacc, fwdlabelacc = makelosses(genatr, settings["model"]["loss"])
Gloss = @λ (x,z) -> mean(-log.(discrm(x)) .- log.(1 .- discrm(genatr(z))))
Dloss = @λ (z) -> mean(-log.(discrm(genatr(z))))

fwdopt = Flux.ADAM(1e-3, (0.9, 0.999))
Gopt = Flux.ADAM(2e-4, (0.5, 0.999))
Dopt = Flux.ADAM(2e-4, (0.5, 0.999))
fwdlrfun(e) = MWFLearning.fixedlr(e,fwdopt)
Glrfun(e) = MWFLearning.fixedlr(e,fwdopt)
Dlrfun(e) = MWFLearning.fixedlr(e,fwdopt)

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
    labelnames = permutedims(settings["data"]["labels"]) # .* " (" .* settings["plot"]["units"] .* ")"
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
            currloss, curracc, currlaberr = mean([Flux.cpu(Flux.data(fwdloss(b...))) for b in train_fwd]), mean([Flux.cpu(Flux.data(fwdacc(b...))) for b in train_fwd]), mean([Flux.cpu(Flux.data(fwdlabelacc(b...))) for b in train_fwd])
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
            currloss, curracc, currlaberr = Flux.cpu(Flux.data(fwdloss(test_fwd...))), Flux.cpu(Flux.data(fwdacc(test_fwd...))), Flux.cpu(Flux.data(fwdlabelacc(test_fwd...)))
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
    save_time = @elapsed let fwdopt = MWFLearning.opt_to_cpu(fwdopt, Flux.params(genatr)), genatr = Flux.cpu(genatr)
        savebson("models/" * FILE_PREFIX * "genatr-checkpoint.bson", @dict(genatr, fwdopt))
    end
    @info @sprintf("[%d] -> Model checkpoint... (%d ms)", CB_EPOCH, 1000 * save_time)
end
checkpoint_errs_cb = function()
    save_time = @elapsed let errs = deepcopy(errs)
        savebson("log/" * FILE_PREFIX * "errors.bson", @dict(errs))
    end
    @info @sprintf("[%d] -> Error checkpoint... (%d ms)", CB_EPOCH, 1000 * save_time)
end

gencbs = Flux.Optimise.runall([
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
        last_lr = lr(fwdopt)
        curr_lr = lr!(fwdopt, fwdlrfun(epoch))
        
        (epoch == 1)         &&  @info(" -> Initial learning rate: " * @sprintf("%.2e", curr_lr))
        (last_lr != curr_lr) &&  @info(" -> Learning rate updated: " * @sprintf("%.2e", last_lr) * " --> "  * @sprintf("%.2e", curr_lr))
        (lr(fwdopt) < 1e-6)  && (@info(" -> Early-exiting: Learning rate has dropped below 1e-6"); break)

        # Train supervised generator loss for one epoch
        train_time = @elapsed Flux.train!(fwdloss, Flux.params(genatr), train_fwd, fwdopt; cb = gencbs) # CuArrays.@sync
        acc_time = @elapsed begin
            acc = Flux.cpu(Flux.data(fwdacc(test_fwd...))) # CuArrays.@sync
            push!(loop_errs[:testing][:epoch], epoch)
            push!(loop_errs[:testing][:acc], acc)
        end
        @info @sprintf("[%d] (%d ms):         Label accuracy: %.4f (%d ms)", epoch, 1000 * train_time, acc, 1000 * acc_time)

        discrm_train_set = [(sampler(batchsize(features(b))),) for b in train_fwd]
        discrm_test_set  = (sampler(batchsize(features(test_fwd))),)
        genatr_train_set = tuple.(labels.(train_fwd), sampler.(batchsize.(labels.(train_fwd))))
        genatr_test_set = (labels(test_fwd), sampler(batchsize(labels(test_fwd))))
        
        # Discriminator loss
        train_time = @elapsed Flux.train!(Gloss, Flux.params(genatr), genatr_train_set, Gopt) # CuArrays.@sync
        Dloss_time = @elapsed test_Dloss = Flux.cpu(Flux.data(Dloss(discrm_test_set...))) # CuArrays.@sync
        @info @sprintf("[%d] (%d ms): Discriminator accuracy: %.4f (%d ms)", epoch, 1000 * train_time, test_Dloss, 1000 * Dloss_time)
        
        # Generator loss
        train_time = @elapsed Flux.train!(Dloss, Flux.params(discrm), discrm_train_set, Dopt) # CuArrays.@sync
        Gloss_time = @elapsed test_Gloss = Flux.cpu(Flux.data(Gloss(genatr_test_set...))) # CuArrays.@sync
        @info @sprintf("[%d] (%d ms):     Generator accuracy: %.4f (%d ms)", epoch, 1000 * train_time, test_Gloss, 1000 * Gloss_time)
        
        # If this is the best accuracy we've seen so far, save the model out
        if acc >= BEST_ACC
            BEST_ACC = acc
            LAST_IMPROVED_EPOCH = epoch
            try
                save_time = @elapsed let weights = Flux.cpu.(Flux.data.(Flux.params(genatr)))
                    savebson("weights/" * FILE_PREFIX * "weights.bson", @dict(weights, epoch, acc))
                end
                # @info " -> New best accuracy; weights saved ($(round(1000*save_time; digits = 2)) ms)"
                @info @sprintf("[%d] -> New best accuracy; weights saved (%d ms)", epoch, 1000 * save_time)
            catch e
                @warn "Error saving weights"
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
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
true_signals   = labels(test_fwd)   |> Flux.cpu |> deepcopy
true_thetas    = features(test_fwd) |> Flux.cpu |> deepcopy
genatr_signals = labels(test_fwd)   |> Flux.data |> Flux.cpu |> deepcopy
genatr_thetas  = genatr(features(test_fwd)) |> Flux.data |> Flux.cpu |> deepcopy
if settings["model"]["problem"] == "forward"; genatr_signals, genatr_thetas = genatr_thetas, genatr_signals; end

error("got here")

prediction_hist = function()
    pred_hist = function(i)
        scale = settings["plot"]["scale"][i]
        units = settings["plot"]["units"][i]
        err = scale .* (genatr_thetas[i,:] .- true_thetas[i,:])
        histogram(err;
            grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["labels"][i] * " ($units)",
            title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
    end
    plot([pred_hist(i) for i in 1:size(genatr_thetas, 1)]...)
end
@info "Plotting prediction histograms..."
fig = prediction_hist()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "labelhistograms.png")

prediction_scatter = function()
    pred_scatter = function(i)
        scale = settings["plot"]["scale"][i]
        units = settings["plot"]["units"][i]
        datascale = scale * settings["model"]["scale"][i]
        p = scatter(scale * true_thetas[i,:], scale * genatr_thetas[i,:];
            marker = :circle, grid = true, minorgrid = true, titlefontsize = 10,
            label = settings["data"]["labels"][i] * " ($units)",
            # title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
        plot!(p, identity, ylims(p)...; line = (:dash, 2, :red), label = L"y = x")
    end
    plot([pred_scatter(i) for i in 1:size(genatr_thetas, 1)]...)
end
@info "Plotting prediction scatter plots..."
fig = prediction_scatter()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "labelscatter.png")

forward_plot = function()
    forward_rmse = function(i)
        y = sum(signals(test_fwd)[:,1,:,i]; dims = 2) # Assumes signal is split linearly into channels
        z_class = MWFLearning.myelin_prop(true_thetas[:,i]...) # Forward simulation of true parameters
        z_genatr =
            settings["model"]["problem"] == "forward" ? genatr_signals : # Forward simulated signal from trained genatr
            MWFLearning.myelin_prop(genatr_thetas[:,i]...) # Forward simulation of genatr predicted parameters
        return (e_class = rmsd(y, z_class), e_genatr = rmsd(y, z_genatr))
    end
    errors = [forward_rmse(i) for i in 1:batchsize(features(test_fwd))]
    e_class = (e -> e.e_class).(errors)
    e_genatr = (e -> e.e_genatr).(errors)
    p = scatter([e_class e_genatr];
        labels = ["RMSE: Classical" "RMSE: Model"],
        marker = [:circle :square],
        grid = true, minorgrid = true, titlefontsize = 10, ylim = (0, 0.05)
    )
end
@info "Plotting forward simulation error plots..."
fig = forward_plot()
display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "forwarderror.png")

errorvslr = function()
    x = [fwdlrfun.(errs[:training][:epoch]), fwdlrfun.(errs[:testing][:epoch])]
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
    err = genatr_thetas .- true_thetas
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