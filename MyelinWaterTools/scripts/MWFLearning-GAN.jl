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

flatsignal = x -> reshape(x, :, batchsize(x))
train_fwd = training_batches(data_set[:training_thetas], flatsignal(data_set[:training_data]), settings["data"]["batch_size"])
train_inv = training_batches(data_set[:training_data], data_set[:training_thetas], settings["data"]["batch_size"])
test_fwd = testing_batches(data_set[:testing_thetas], flatsignal(data_set[:testing_data]))
test_inv = testing_batches(data_set[:testing_data], data_set[:testing_thetas])

real_train_set() = tuple.(labels.(train_fwd))
fake_train_set() = tuple.(sampler.(batchsize.(features.(train_fwd))))
pair_train_set() = tuple.(labels.(train_fwd), sampler.(batchsize.(features.(train_fwd))))
real_test_set()  = tuple(labels(test_fwd))
fake_test_set()  = tuple(sampler(batchsize(features(test_fwd))))
pair_test_set()  = tuple(labels(test_fwd), sampler(batchsize(features(test_fwd))))

# Construct model
@info "Constructing model..."
genatr, discrm, sampler = MWFLearning.get_model(settings);
genatr = GPU ? Flux.gpu(genatr) : genatr;
discrm = GPU ? Flux.gpu(discrm) : discrm;
Dparams, Gparams = Flux.params(discrm), Flux.params(genatr)
DGparams = Flux.Tracker.Params([collect(Dparams); collect(Gparams)])

model_summary(genatr, joinpath(savefolders["models"], FILE_PREFIX * "generator.architecture.txt"));
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
Gloss = @λ (z) -> mean(.-log.(discrm(genatr(z))))
Dloss = @λ (x,z) -> mean(.-log.(discrm(x)) .- log.(1 .- discrm(genatr(z))))
realDloss = @λ (x) -> mean(.-log.(discrm(x)))
fakeDloss = @λ (z) -> mean(.-log.(1 .- discrm(genatr(z))))

fwdopt   = Flux.ADAM(1e-3, (0.9, 0.999))
Gopt     = Flux.ADAM(2e-4, (0.5, 0.999))
Dopt     = Flux.ADAM(2e-4, (0.5, 0.999))
realDopt = Flux.ADAM(2e-4, (0.5, 0.999))
fakeDopt = Flux.ADAM(2e-4, (0.5, 0.999))
fwdlrfun(e) = MWFLearning.fixedlr(e,fwdopt)
Glrfun(e) = MWFLearning.fixedlr(e,fwdopt)
Dlrfun(e) = MWFLearning.fixedlr(e,fwdopt)

# Global training state, accumulators, etc.
state = Dict(
    :epoch                => 0,
    :best_acc             => 0.0,
    :last_improved_epoch  => 0,
    :acc_thresh           => 100.0, # Never stop
    :drop_lr_thresh       => typemax(Int), # Drop step size after this many stagnant epochs
    :converged_thresh     => typemax(Int), # Call model converged after this many stagnant epochs
    :loop => Dict( # Values which are updated within the training loop explicitly
        :epoch => Int[], :acc => T[], :Gloss => T[], :Dloss => T[], :D_x => T[], :D_G_z => T[]),
    :callbacks => Dict( # Values which are updated within callbacks (should not be touched in training loop)
        :training => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]),
        :testing => Dict(:epoch => Int[], :loss => T[], :acc => T[], :labelerr => VT[]))
)

update_lr_cb            = MWFLearning.make_update_lr_cb(state, fwdopt, fwdlrfun)
test_err_cb             = MWFLearning.make_test_err_cb(state, fwdloss, fwdacc, fwdlabelacc, test_fwd)
train_err_cb            = MWFLearning.make_train_err_cb(state, fwdloss, fwdacc, fwdlabelacc, train_fwd)
plot_errs_cb            = MWFLearning.make_plot_errs_cb(state, "plots/" * FILE_PREFIX * "errs.png"; labelnames = permutedims(settings["data"]["labels"]), labellegend = (settings["model"]["problem"] == "forward" ? nothing : :topleft))
plot_gan_losses_cb      = MWFLearning.make_plot_gan_losses_cb(state, "plots/" * FILE_PREFIX * "ganloss.png")
checkpoint_state_cb     = MWFLearning.make_checkpoint_state_cb(state, "log/" * FILE_PREFIX * "errors.bson")
save_best_model_cb      = MWFLearning.make_save_best_model_cb(state, genatr, "weights/" * FILE_PREFIX * "weights.bson")
checkpoint_model_opt_cb = MWFLearning.make_checkpoint_model_opt_cb(state, genatr, fwdopt, "models/" * FILE_PREFIX * "genatr-checkpoint.bson")

pretraincbs = Flux.Optimise.runall([
    update_lr_cb,
])

posttraincbs = Flux.Optimise.runall([
    epochthrottle(test_err_cb, state, 5),
    epochthrottle(train_err_cb, state, 5),
    epochthrottle(plot_errs_cb, state, 5),
    Flux.throttle(checkpoint_state_cb, 60),
    # Flux.throttle(checkpoint_model_opt_cb, 120),
])

loopcbs = Flux.Optimise.runall([
    save_best_model_cb,
    epochthrottle(plot_gan_losses_cb, state, 5),
])

# Training Loop
train_loop! = function()
    for epoch in state[:epoch] .+ (1:settings["optimizer"]["epochs"])
        state[:epoch] = epoch
        
        # Call pre-training callbacks
        pretraincbs()

        # Supervised generator training
        train_time = @elapsed Flux.train!(fwdloss, Gparams, train_fwd, fwdopt) # CuArrays.@sync
        acc_time = @elapsed acc = Flux.cpu(Flux.data(fwdacc(test_fwd...))) # CuArrays.@sync
        @info @sprintf("[%d] (%4d ms): Label accuracy: %.4f (%d ms)", epoch, 1000 * train_time, acc, 1000 * acc_time)

        # Call post-training callbacks
        posttraincbs()

        # Generator training
        train_time = @elapsed Flux.train!(Gloss, Gparams, fake_train_set(), Gopt) # CuArrays.@sync
        Gloss_time = @elapsed test_Gloss = Flux.cpu(Flux.data(Gloss(fake_test_set()...))) # CuArrays.@sync
        @info @sprintf("[%d] (%4d ms): Generator loss: %.4f (%d ms)", epoch, 1000 * train_time, test_Gloss, 1000 * Gloss_time)

        # # Discriminator training
        # train_time = @elapsed Flux.train!(Dloss, Dparams, pair_train_set(), Dopt) # CuArrays.@sync
        # Dloss_time = @elapsed test_Dloss = Flux.cpu(Flux.data(Dloss(pair_test_set()...))) # CuArrays.@sync
        # @info @sprintf("[%d] (%4d ms):   Discrim loss: %.4f (%d ms)", epoch, 1000 * train_time, test_Dloss, 1000 * Dloss_time)
        
        # Alternating discriminator training
        train_time = @elapsed for _ in 1:3
            all_train_time  = @elapsed Flux.train!(Dloss, Dparams, pair_train_set(), Dopt) # CuArrays.@sync
            # real_train_time = @elapsed Flux.train!(realDloss, Dparams, real_train_set(), realDopt) # CuArrays.@sync
            # fake_train_time = @elapsed Flux.train!(fakeDloss, Dparams, fake_train_set(), fakeDopt) # CuArrays.@sync
        end
        Dloss_time = @elapsed test_Dloss = Flux.cpu(Flux.data(Dloss(pair_test_set()...))) # CuArrays.@sync
        @info @sprintf("[%d] (%4d ms):   Discrim loss: %.4f (%d ms)", epoch, 1000 * train_time, test_Dloss, 1000 * Dloss_time)
        
        # Discriminator performance
        Dperf_time = @elapsed D_x, D_G_z = mean(Flux.data(discrm(real_test_set()...))), mean(Flux.data(discrm(genatr(fake_test_set()...))))
        @info @sprintf("[%d] (%4d ms):  D(x), D(G(z)): %.4f, %.4f", epoch, 1000 * Dperf_time, D_x, D_G_z)
        
        # Update loop values
        push!(state[:loop][:epoch], epoch)
        push!(state[:loop][:acc], acc)
        push!(state[:loop][:Gloss], test_Gloss)
        push!(state[:loop][:Dloss], test_Dloss)
        push!(state[:loop][:D_x], D_x)
        push!(state[:loop][:D_G_z], D_G_z)
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
    x = [fwdlrfun.(state[:callbacks][:training][:epoch]), fwdlrfun.(state[:callbacks][:testing][:epoch])]
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