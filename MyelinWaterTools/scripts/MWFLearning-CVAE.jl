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
pyplot(size=(800,600))

# Settings
const settings_file = "settings.toml"
const settings = TOML.parsefile(settings_file)

const DATE_PREFIX = getnow() * "."
const FILE_PREFIX = DATE_PREFIX * model_string(settings) * "."
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
maybegpu(x) = GPU ? Flux.gpu(x) : x
for k in (:training_data, :testing_data, :training_thetas, :testing_thetas); data_set[k] = maybegpu(data_set[k]); end

BT_train_data = training_batches(data_set[:training_thetas], data_set[:training_data], settings["data"]["train_batch"])
BT_test_data = testing_batches(data_set[:testing_thetas], data_set[:testing_data])

const thetas_infer_idx = 1:length(settings["data"]["info"]["labinfer"])
thetas = batch -> features(batch)[thetas_infer_idx, :]
signals = batch -> labels(batch)
flipbatch(batch) = (signals(batch), thetas(batch))

# Lazy data loader for training on simulated data
const mb_sampler_batch_size  = 250
const mb_sampler_train_batch = 10
function x_sampler()
    out = zeros(T, 8, mb_sampler_batch_size)
    for j in 1:size(out,2)
        mwf    = MWFLearning.linearsampler(0.15, 0.3)
        ewf    = MWFLearning.linearsampler(0.29, 0.32)
        iwf    = 1 - (mwf + ewf)
        alpha  = MWFLearning.linearsampler(120.0, 180.0)
        K      = MWFLearning.log10sampler(1e-3, 1.0)
        T2sp   = MWFLearning.linearsampler(10e-3, 20e-3)
        T2lp   = MWFLearning.linearsampler(50e-3, 80e-3)
        T2tiss = MWFLearning.linearsampler(50e-3, 80e-3)
        T1sp   = MWFLearning.linearsampler(150e-3, 250e-3)
        T1lp   = MWFLearning.linearsampler(949e-3, 1219e-3)
        T1tiss = MWFLearning.linearsampler(949e-3, 1219e-3)
        TE     = 10e-3
        out[1,j] = mwf
        out[2,j] = 1-mwf # iewf
        out[3,j] = inv((iwf * inv(T2lp) + ewf * inv(T2tiss)) / (iwf + ewf)) / TE # T2iew/TE
        out[4,j] = T2sp / TE # T2mw/TE
        out[5,j] = alpha
        out[6,j] = log10(TE*K)
        out[7,j] = inv((iwf * inv(T1lp) + ewf * inv(T1tiss)) / (iwf + ewf)) / TE # T1iew/TE
        out[8,j] = T1sp / TE # T1mw/TE
    end
    return out
end
y_sampler(x) = (y = MWFLearning.forward_physics(x); reshape(y, size(y,1), 1, 1, :))
mb_sampler = MWFLearning.LazyMiniBatches(mb_sampler_train_batch, x_sampler, y_sampler)

# Construct model
@info "Constructing model..."
@unpack m = MWFLearning.make_model(settings)[1] |> maybegpu;
model_summary(m, joinpath(savefolders["models"], FILE_PREFIX * "architecture.txt"));
param_summary(m, flipbatch.(BT_train_data), flipbatch(BT_test_data));

# Loss and accuracy function
theta_weights()::CVT = inv.(settings["data"]["info"]["labwidth"][thetas_infer_idx]) .* unitsum(settings["data"]["info"]["labweights"]) |> copy |> VT |> maybegpu |> CVT
datanoise = (snr = T(settings["data"]["postprocess"]["SNR"]); snr ≤ 0 ? identity : @λ(y -> MWFLearning.add_rician(y, snr)))
trainloss = @λ (x,y) -> MWFLearning.H_LIGOCVAE(m, x, datanoise(y))
θloss, θacc, θerr = make_losses(@λ(y -> m(datanoise(y); nsamples = 10)), settings["model"]["loss"], theta_weights())

# Optimizer
opt = Flux.ADAM(settings["optimizer"]["ADAM"]["lr"], (settings["optimizer"]["ADAM"]["beta"]...,))
lrfun(e) = MWFLearning.fixedlr(e,opt)

# Global training state, accumulators, etc.
state = Dict(
    :epoch                => 0,
    :best_acc             => 0.0,
    :last_improved_epoch  => 0,
    :acc_thresh           => 100.0, # Never stop
    :drop_lr_thresh       => typemax(Int), # Drop step size after this many stagnant epochs
    :converged_thresh     => typemax(Int), # Call model converged after this many stagnant epochs
    :loop => Dict( # Values which are updated within the training loop explicitly
        :epoch => Int[], :acc => T[], :loss => T[], :ELBO => T[], :KL => T[]),
    :callbacks => Dict( # Values which are updated within callbacks (should not be touched in training loop)
        :training => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]),
        :testing => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]))
)

update_lr_cb            = MWFLearning.make_update_lr_cb(state, opt, lrfun)
test_err_cb             = MWFLearning.make_test_err_cb(state, θloss, θacc, θerr, flipbatch(BT_test_data))
train_err_cb            = MWFLearning.make_train_err_cb(state, θloss, θacc, θerr, flipbatch.(BT_train_data))
plot_errs_cb            = MWFLearning.make_plot_errs_cb(state, "plots/" * FILE_PREFIX * "errs.png"; labelnames = permutedims(settings["data"]["info"]["labinfer"]))
plot_ligocvae_losses_cb = MWFLearning.make_plot_ligocvae_losses_cb(state, "plots/" * FILE_PREFIX * "ligocvae.png")
checkpoint_state_cb     = MWFLearning.make_checkpoint_state_cb(state, "log/" * FILE_PREFIX * "errors.bson")
checkpoint_model_opt_cb = MWFLearning.make_checkpoint_model_opt_cb(state, m, opt, "models/" * FILE_PREFIX * "model-checkpoint.bson")
save_best_model_cb      = MWFLearning.make_save_best_model_cb(state, m, "weights/" * FILE_PREFIX * "weights.bson")

pretraincbs = Flux.Optimise.runall([
    update_lr_cb,
])

posttraincbs = Flux.Optimise.runall([
    epochthrottle(test_err_cb, state, 25),
    epochthrottle(train_err_cb, state, 25),
    epochthrottle(plot_errs_cb, state, 25),
    epochthrottle(checkpoint_state_cb, state, 25),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
])

loopcbs = Flux.Optimise.runall([
    save_best_model_cb,
    epochthrottle(plot_ligocvae_losses_cb, state, 25),
])

# Train using Bloch-Torrey training/testing data, or sampler data
train_data, test_data = BT_train_data, BT_test_data
# train_data, test_data = mb_sampler, rand(mb_sampler)

# Training Loop
train_loop! = function()
    for epoch in state[:epoch] .+ (1:settings["optimizer"]["epochs"])
        state[:epoch] = epoch
        
        pretraincbs() # pre-training callbacks
        train_time = @elapsed Flux.train!(trainloss, Flux.params(m), train_data, opt) # CuArrays.@sync
        acc_time = @elapsed acc = θacc(flipbatch(test_data)...) |> Flux.data |> Flux.cpu # CuArrays.@sync
        posttraincbs() # post-training callbacks
        
        @info @sprintf("[%d] (%4d ms): Label accuracy: %.4f (%d ms)", epoch, 1000 * train_time, acc, 1000 * acc_time)
        
        # Update loop values
        push!(state[:loop][:epoch], epoch)
        push!(state[:loop][:acc], acc)
        push!(state[:loop][:loss], MWFLearning.H_LIGOCVAE(m, test_data...) |> Flux.data)
        push!(state[:loop][:ELBO], MWFLearning.L_LIGOCVAE(m, test_data...) |> Flux.data)
        push!(state[:loop][:KL], MWFLearning.KL_LIGOCVAE(m, test_data...) |> Flux.data)
        loopcbs()
    end
end

@info("Beginning training loop...")
try
    train_loop!()
catch e
    if e isa InterruptException
        @info "Training interrupted by user; breaking out of loop..."
    elseif e isa Flux.Optimise.StopException
        @info "Training stopped by callback..."
    else
        @warn "Error during training..."
        @warn sprint(showerror, e, catch_backtrace())
    end
end

@info "Computing resulting labels..."
map((p, pbest) -> Flux.data(p) .= pbest, Flux.params(m), BSON.load("weights/" * FILE_PREFIX * "weights.bson")[:weights]);
true_signals = signals(BT_test_data) |> Flux.cpu;
true_thetas  = thetas(BT_test_data) |> Flux.cpu;
model_thetas = m(signals(BT_test_data); nsamples = 1000) |> Flux.cpu;
model_stds   = std([m(signals(BT_test_data); nsamples = 1) |> Flux.cpu for _ in 1:1000]);

prediction_hist = function()
    pred_hist = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        err = scale .* (model_thetas[i,:] .- true_thetas[i,:])
        histogram(err;
            grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
            title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
    end
    plot([pred_hist(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting prediction histograms..."
fig = prediction_hist(); display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "theta.histogram.png")

prediction_scatter = function()
    pred_scatter = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        p = scatter(scale .* true_thetas[i,:], scale .* model_thetas[i,:];
            marker = :circle, grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
            # title = "|μ| = $(round(mean(abs.(err)); sigdigits = 2)), μ = $(round(mean(err); sigdigits = 2)), σ = $(round(std(err); sigdigits = 2))", #, IQR = $(round(iqr(err); sigdigits = 2))",
        )
        plot!(p, identity, ylims(p)...; line = (:dash, 2, :red), label = L"y = x")
    end
    plot([pred_scatter(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting prediction scatter plots..."
fig = prediction_scatter(); display(fig)
savefig(fig, "plots/" * FILE_PREFIX * "theta.scatter.png")

forward_plot = function()
    forward_rmse = function(i)
        y = sum(signals(BT_test_data)[:,1,:,i]; dims = 2) # Assumes signal is split linearly into channels
        z_class = MWFLearning.forward_physics(true_thetas[:,i:i]) # Forward simulation of true parameters
        z_model = MWFLearning.forward_physics(model_thetas[:,i:i]) # Forward simulation of model predicted parameters
        return (e_class = rmsd(y, z_class), e_model = rmsd(y, z_model))
    end
    errors = [forward_rmse(i) for i in 1:batchsize(thetas(BT_test_data))]
    e_class = (e -> e.e_class).(errors)
    e_model = (e -> e.e_model).(errors)
    p = scatter([e_class e_model];
        labels = ["RMSE: Classical" "RMSE: Model"],
        marker = [:circle :square],
        grid = true, minorgrid = true, titlefontsize = 10, ylim = (0, 0.05)
    )
end
# @info "Plotting forward simulation error plots..."
# fig = forward_plot()
# display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "forwarderror.png")

errorvslr = function()
    x = [lrfun.(state[:callbacks][:training][:epoch]), lrfun.(state[:callbacks][:testing][:epoch])]
    y = [state[:callbacks][:training][:loss], state[:callbacks][:testing][:loss]]
    plot(
        plot(x, (e -> log10.(e .- minimum(e) .+ 1e-6)).(y); xscale = :log10, ylabel = "stretched loss ($(settings["model"]["loss"]))", label = ["training" "testing"]),
        plot(x, (e -> log10.(e)).(y); xscale = :log10, xlabel = "learning rate", ylabel = "loss ($(settings["model"]["loss"]))", label = ["training" "testing"]);
        layout = (2,1)
    )
end
# @info "Plotting errors vs. learning rate..."
# fig = errorvslr()
# display(fig) && savefig(fig, "plots/" * FILE_PREFIX * "lossvslearningrate.png")

mwferrorvsthetas = function()
    mwf_err = (model_thetas .- true_thetas)[1,:]
    sp = data_set[:testing_data_dicts][1][:sweepparams]
    for k in keys(sp)
        xdata = (d -> d[:sweepparams][k]).(data_set[:testing_data_dicts])
        p1 = scatter(xdata, abs.(mwf_err); xlabel = string(k), ylabel = "|mwf error|")
        p2 = scatter(xdata, mwf_err; xlabel = string(k), ylabel = "mwf error")
        p = plot(p1, p2; layout = (1,2), m = (10, :c))
        display(p)
    end
end
# @info "Plotting mwf error vs. thetas..."
# mwferrorvsthetas()
