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

const SAVE = true
const DATE_PREFIX = getnow() * "."
const FILE_PREFIX = DATE_PREFIX * model_string(settings) * "."
const GPU = settings["gpu"] :: Bool
const T   = settings["prec"] == 64 ? Float64 : Float32
const VT  = Vector{T}
const MT  = Matrix{T}
const VMT = VecOrMat{T}
const CVT = GPU ? CuVector{T} : Vector{T}
maybegpu(x) = GPU ? Flux.gpu(x) : x
savepath(folder, filename) = SAVE ? joinpath(savefolders[folder], FILE_PREFIX * filename) : nothing

const savefoldernames = ["settings", "models", "log", "plots"]
const savefolders = Dict{String,String}(savefoldernames .=> mkpath.(joinpath.(settings["dir"], savefoldernames)))
SAVE && cp(settings_file, savepath("settings", "settings.toml"); force = true)

# Load and prepare signal data
@info "Preparing data..."
const data_set = prepare_data(settings)
map(k -> data_set[k] = maybegpu(data_set[k]), (:training_data, :testing_data, :training_thetas, :testing_thetas))

const BT_train_data = training_batches(data_set[:training_thetas], data_set[:training_data], settings["data"]["train_batch"])
const BT_test_data = testing_batches(data_set[:testing_thetas], data_set[:testing_data])

const thetas_infer_idx = 1:length(settings["data"]["info"]["labinfer"]) # thetas to learn during parameter inference
thetas = batch -> features(batch)[thetas_infer_idx, :]
signals = batch -> labels(batch)
labelbatch(batch) = (signals(batch), thetas(batch))

# Lazy data loader for training on simulated data
const MB_train_batch_size  = 10_000 # Number of simulated signals in one training batch
const MB_test_batch_size   = 500    # Number of simulated signals in one testing batch
const MB_num_train_batches = 10     # Number of training batches per epoch
function x_sampler()
    out = zeros(T, 14, MB_train_batch_size)
    for j in 1:size(out,2)
        g_bounds, η_bounds = (0.60, 0.92), (0.15, 0.82)
        mvf    = MWFLearning.linearsampler(0.025, 0.4)
        η      = min(mvf / (1 - g_bounds[2]^2), η_bounds[2]) # Maximum density for given bounds
        g      = sqrt(1 - mvf/η) # g-ratio corresponding to maximum density
        # g      = MWFLearning.linearsampler(g_bounds...) # Uniformly random g-ratio
        # η      = mvf / (1 - g^2) # MWFLearning.linearsampler(η_bounds...)
        evf    = 1 - η
        ivf    = 1 - (mvf + evf)
        mwf    =  mvf / (2evf + 2ivf + mvf)
        ewf    = 2evf / (2evf + 2ivf + mvf)
        iwf    = 2ivf / (2evf + 2ivf + mvf)
        alpha  = MWFLearning.linearsampler(120.0, 180.0)
        K      = MWFLearning.log10sampler(1e-3, 10.0)
        T2sp   = MWFLearning.linearsampler(10e-3, 70e-3)
        T2lp   = MWFLearning.linearsampler(50e-3, 180e-3)
        while !(T2lp ≥ 1.5*T2sp) # Enforce constraint
            T2sp = MWFLearning.linearsampler(10e-3, 70e-3)
            T2lp = MWFLearning.linearsampler(50e-3, 180e-3)
        end
        T1sp   = MWFLearning.linearsampler(150e-3, 250e-3)
        T1lp   = MWFLearning.linearsampler(949e-3, 1219e-3)
        T2tiss = T2lp # MWFLearning.linearsampler(50e-3, 180e-3)
        T1tiss = T1lp # MWFLearning.linearsampler(949e-3, 1219e-3)
        TE     = 10e-3
        out[1,j]  = cosd(alpha) # cosd(alpha)
        out[2,j]  = g # gratio
        out[3,j]  = mwf # mwf
        out[4,j]  = T2sp / TE # T2mw/TE
        out[5,j]  = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        out[6,j]  = iwf # iwf
        out[7,j]  = ewf # ewf
        out[8,j]  = iwf + ewf # iewf
        out[9,j]  = T2lp / TE # T2iw/TE
        out[10,j] = T2tiss / TE # T2ew/TE
        out[11,j] = T1sp / TE # T1mw/TE
        out[12,j] = T1lp / TE # T1iw/TE
        out[13,j] = T1tiss / TE # T1ew/TE
        out[14,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
        #out[1,j] = mwf
        #out[2,j] = 1 - mwf # iewf
        #out[3,j] = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        #out[4,j] = T2sp / TE # T2mw/TE
        #out[5,j] = alpha
        #out[6,j] = log10(TE*K)
        #out[7,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
        #out[8,j] = T1sp / TE # T1mw/TE
    end
    return out
end
# y_sampler(x) = (y = MWFLearning.forward_physics_8arg(x); reshape(y, size(y,1), 1, 1, :))
y_sampler(x) = (y = MWFLearning.forward_physics_14arg(x); reshape(y, size(y,1), 1, 1, :))
MB_sampler = MWFLearning.LazyMiniBatches(MB_num_train_batches, x_sampler, y_sampler)

# Train using Bloch-Torrey training/testing data, or sampler data
# train_data, test_data = BT_train_data, BT_test_data
train_data, test_data = MB_sampler, (x -> x[.., 1:MB_test_batch_size]).(rand(MB_sampler))

# Construct model
@info "Constructing model..."
# @unpack m = MWFLearning.make_model(settings)[1] |> maybegpu;
m = BSON.load("log/2019-10-07-T-18-45-09-415.acc=rmse_loss=l2_DenseLIGOCVAE_Ndense1=128_Ndense2=128_Ndense3=128_Ndense4=128_Xout=5_Zdim=20_act=leakyrelu.model-best.bson")[:model] |> deepcopy
model_summary(m, savepath("models", "architecture.txt"));
param_summary(m, labelbatch.(train_data), labelbatch(test_data));

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
test_err_cb             = MWFLearning.make_test_err_cb(state, θloss, θacc, θerr, labelbatch(test_data))
train_err_cb            = MWFLearning.make_train_err_cb(state, θloss, θacc, θerr, labelbatch.(train_data))
plot_errs_cb            = MWFLearning.make_plot_errs_cb(state, savepath("plots", "errs.png"); labelnames = permutedims(settings["data"]["info"]["labinfer"]))
plot_ligocvae_losses_cb = MWFLearning.make_plot_ligocvae_losses_cb(state, savepath("plots", "ligocvae.png"))
checkpoint_state_cb     = MWFLearning.make_checkpoint_state_cb(state, savepath("log", "errors.bson"))
save_best_model_cb      = MWFLearning.make_save_best_model_cb(state, m, opt, savepath("log", "")) # suffix set internally
checkpoint_model_cb     = MWFLearning.make_checkpoint_model_cb(state, m, opt, savepath("log", "")) # suffix set internally

pretraincbs = Flux.Optimise.runall([
    update_lr_cb,
])

posttraincbs = Flux.Optimise.runall([
    epochthrottle(test_err_cb, state, 25),
    epochthrottle(train_err_cb, state, 25),
    epochthrottle(plot_errs_cb, state, 25),
])

loopcbs = Flux.Optimise.runall([
    save_best_model_cb,
    epochthrottle(plot_ligocvae_losses_cb, state, 25),
    epochthrottle(checkpoint_state_cb, state, 25),
    epochthrottle(checkpoint_model_cb, state, 25),
])

# Training Loop
train_loop! = function()
    for epoch in state[:epoch] .+ (1:settings["optimizer"]["epochs"])
        state[:epoch] = epoch
        
        pretraincbs() # pre-training callbacks
        train_time = @elapsed Flux.train!(trainloss, Flux.params(m), train_data, opt) # CuArrays.@sync
        acc_time = @elapsed acc = θacc(labelbatch(test_data)...) |> Flux.data |> Flux.cpu # CuArrays.@sync
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
best_model   = SAVE ? BSON.load(savepath("log", "model-best.bson"))[:model] : deepcopy(m);
true_thetas  = thetas(test_data) |> Flux.cpu;
true_signals = signals(test_data) |> Flux.cpu;
model_thetas = best_model(signals(test_data); nsamples = 1000) |> Flux.cpu;
# model_stds = std([best_model(signals(test_data); nsamples = 1) |> Flux.cpu for _ in 1:1000]);

# keep_indices = true_thetas[2,:] .≤ -3.5 # Keep small permeability only
# true_thetas  = true_thetas[.., keep_indices]
# true_signals = true_signals[.., keep_indices]
# model_thetas = model_thetas[.., keep_indices]
# # model_stds = model_stds[.., keep_indices]

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
SAVE && savefig(fig, savepath("plots", "theta.histogram.png"))

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
SAVE && savefig(fig, savepath("plots", "theta.scatter.png"))

prediction_corrplot = function()
    # cosalpha(x) = (out = copy(x); @views out[1,:] .= cosd.(out[1,:]); out)
    θ = true_thetas .* settings["data"]["info"]["labscale"] |> permutedims
    Δ = abs.(model_thetas .- true_thetas) .* settings["data"]["info"]["labscale"] |> permutedims
    # θlabs = settings["data"]["info"]["labinfer"] .* " [" .* settings["data"]["info"]["labunits"] .* "]" |> permutedims
    θlabs = settings["data"]["info"]["labinfer"] |> permutedims
    Δlabs = θlabs .* " error"
    θidx = 1:2
    Δidx = 1:length(Δlabs)
    corrdata = hcat(θ[..,θidx], Δ[..,Δidx])
    corrlabs = hcat(θlabs[..,θidx], Δlabs[..,Δidx])
    fig = corrplot(corrdata; label = corrlabs, fillcolor = :purple, markercolor = cgrad(:rainbow, :misc), xrotation = 45.0, guidefontsize = 8, tickfontsize = 8)
    # savefig(fig, "tmp.pdf")
end
@info "Plotting correlation plots..."
fig = prediction_corrplot(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.corrplot.png"))

forward_plot = function()
    forward_rmse = function(i)
        y = sum(signals(test_data)[:,1,:,i]; dims = 2) # Assumes signal is split linearly into channels
        z_class = MWFLearning.forward_physics_14arg(true_thetas[:,i:i]) # Forward simulation of true parameters
        z_model = MWFLearning.forward_physics_14arg(model_thetas[:,i:i]) # Forward simulation of model predicted parameters
        return (e_class = rmsd(y, z_class), e_model = rmsd(y, z_model))
    end
    errors = [forward_rmse(i) for i in 1:batchsize(thetas(test_data))]
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
# display(fig) && savefig(fig, savepath("plots", "forwarderror.png"))

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
# display(fig) && savefig(fig, savepath("plots", "lossvslearningrate.png"))

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
