# Initialization project/code loading
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics: mean, median, std
using StatsBase: quantile, sample, iqr
using Base.Iterators: repeated, partition
using Printf

using MWFLearning
# using CuArrays
pyplot(size=(800,600))
# pyplot(size=(1600,900))

# Settings
findendswith(dir, suffix) = filter!(s -> endswith(s, suffix), readdir(dir)) |> x -> isempty(x) ? nothing : joinpath(dir, first(x))
const default_settings_file = "/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MyelinWaterTools/MWFLearning/examples/cvae_settings.toml"
const settings = let
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(default_settings_file)
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
    if haskey(ENV, "SETTINGSFILE")
        merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))
    end
    if false #TODO FIXME
        merge!(mergereducer!, settings, TOML.parsefile(findendswith("/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v5/sweep/18/settings/", ".settings.toml")))
    end
    TOML.print(stdout, settings)
    settings
end

const DATE_PREFIX = getnow() * "."
const FILE_PREFIX = DATE_PREFIX * model_string(settings) * "."
const SCRIPT_TIME_START = Dates.now()
const SCRIPT_TIMEOUT = settings["timeout"] == 0 ? Dates.Second(typemax(Int)) : Dates.Second(ceil(Int, 3600 * settings["timeout"]))
const SAVE = settings["save"] :: Bool
const GPU  = settings["gpu"] :: Bool
const T    = settings["prec"] == 64 ? Float64 : Float32
const VT   = Vector{T}
const MT   = Matrix{T}
const VMT  = VecOrMat{T}
const CVT  = GPU ? CuVector{T} : Vector{T}
maybegpu(x) = GPU ? Flux.gpu(x) : x

const savefoldernames = ["settings", "models", "log", "plots"]
const savefolders = Dict{String,String}(savefoldernames .=> mkpath.(joinpath.(settings["dir"], savefoldernames)))
savepath(folder, filename) = SAVE ? joinpath(savefolders[folder], FILE_PREFIX * filename) : nothing
if SAVE
    # Save + print resulting settings
    open(savepath("settings", "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
end

# Load and prepare signal data
@info "Preparing data..."
const data_set = prepare_data(settings)
map(k -> data_set[k] = maybegpu(data_set[k]), (:training_data, :testing_data, :training_thetas, :testing_thetas))

const BT_train_data = training_batches(data_set[:training_thetas], data_set[:training_data], settings["data"]["train_batch"]) # For overtraining: |> xy -> ((x, y) = xy[1]; [(x[.., 1:100], y[.., 1:100])])
const BT_test_data = testing_batches(data_set[:testing_thetas], data_set[:testing_data])

const thetas_infer_idx = 1:length(settings["data"]["info"]["labinfer"]) # thetas to learn during parameter inference
thetas = batch -> features(batch)[thetas_infer_idx, :]
signals = batch -> labels(batch)
labelbatch(batch) = (signals(batch), thetas(batch))

# Lazy data loader for training on simulated data
#=
const MB_train_batch_size  = 500 # Number of simulated signals in one training batch
const MB_test_batch_size   = 500 # Number of simulated signals in one testing batch
const MB_num_train_batches = 20  # Number of training batches per epoch
x_sampler() = MWFLearning.theta_sampler_8arg(MB_train_batch_size)
y_sampler(x) = reshape(MWFLearning.forward_physics_8arg(x), :, 1, 1, batchsize(x))
MB_sampler = MWFLearning.LazyMiniBatches(MB_num_train_batches, x_sampler, y_sampler)
=#

# Train using Bloch-Torrey training/testing data, or sampler data
train_data, test_data = BT_train_data, BT_test_data
# train_data, test_data = MB_sampler, (x -> x[.., 1:MB_test_batch_size]).(rand(MB_sampler))

# Construct model
@info "Constructing model..."
m = MWFLearning.make_model(settings, "DenseLIGOCVAE") |> maybegpu;
# m = BSON.load(findendswith("/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v5/sweep/18/log/", ".model-checkpoint.bson"))[:model] |> deepcopy; #TODO FIXME
# Flux.testmode!(m, true); #TODO FIXME

model_summary(m, savepath("models", "architecture.txt"));
param_summary(m, labelbatch.(train_data), labelbatch(test_data));

# Loss and accuracy function
theta_weights()::CVT = inv.(settings["data"]["info"]["labwidth"][thetas_infer_idx]) .* unitsum(settings["data"]["info"]["labweights"]) |> copy |> VT |> maybegpu |> CVT
data_noise(y) = (snr = T(settings["data"]["postprocess"]["SNR"]); nrm = settings["data"]["preprocess"]["normalize"]::String; y = snr > 0 ? MWFLearning.add_rician(y, snr) : y; y = nrm == "unitsum" ? unitsum(y; dims = 1) : y; return y)
H_loss  = @λ (x,y) -> MWFLearning.H_LIGOCVAE(m, x, data_noise(y); gamma = T(settings["model"]["gamma"]))
L_loss  = @λ (x,y) -> MWFLearning.L_LIGOCVAE(m, x, data_noise(y))
KL_loss = @λ (x,y) -> MWFLearning.KL_LIGOCVAE(m, x, data_noise(y))
θloss, θacc, θerr = make_losses(@λ(y -> m(data_noise(y); nsamples = 10)), settings["model"]["loss"], theta_weights())

# Optimizer
opt = Flux.ADAMW(
    settings["optimizer"]["ADAM"]["lr"],
    (settings["optimizer"]["ADAM"]["beta"]...,),
    settings["optimizer"]["ADAM"]["decay"],
)
lrfun(e) = clamp(
    MWFLearning.geometriclr(e, opt;
        rate = settings["optimizer"]["lrrate"],
        factor = settings["optimizer"]["lrdrop"],
    ),
    settings["optimizer"]["lrbounds"]...
)

# Global state
timer = TimerOutput()
state = DataFrame(
    :epoch    => Int[], # mandatory field
    :dataset  => Symbol[], # mandatory field
    :loss     => Union{T, Missing}[],
    :acc      => Union{T, Missing}[],
    :ELBO     => Union{T, Missing}[],
    :KL       => Union{T, Missing}[],
    :labelerr => Union{VT, Missing}[],
)
# state = BSON.load(findendswith("/project/st-arausch-1/jcd1994/simulations/ismrm2020/cvae-diff-med-2-v5/sweep/18/log/", ".errors.bson"))[:state] |> deepcopy #TODO FIXME

update_lr_cb            = MWFLearning.make_update_lr_cb(state, opt, lrfun)
checkpoint_state_cb     = MWFLearning.make_checkpoint_state_cb(state, savepath("log", "errors.bson"); filtermissings = true, filternans = true)
checkpoint_model_cb     = MWFLearning.make_checkpoint_model_cb(state, m, opt, savepath("log", "")) # suffix set internally
plot_errs_cb            = MWFLearning.make_plot_errs_cb(state, savepath("plots", "errs.png"); labelnames = permutedims(settings["data"]["info"]["labinfer"]))
plot_ligocvae_losses_cb = MWFLearning.make_plot_ligocvae_losses_cb(state, savepath("plots", "ligocvae.png"))
save_best_model_cb      = MWFLearning.make_save_best_model_cb(state, m, opt, savepath("log", "")) # suffix set internally

pretraincbs = Flux.Optimise.runall([
    update_lr_cb,
])

posttraincbs = Flux.Optimise.runall([
    save_best_model_cb,
    Flux.throttle(checkpoint_state_cb, 300; leading = false), #TODO
    Flux.throttle(checkpoint_model_cb, 300; leading = false), #TODO
    Flux.throttle(plot_errs_cb, 300; leading = false), #TODO
    Flux.throttle(plot_ligocvae_losses_cb, 300; leading = false), #TODO
])

# Training Loop
train_loop! = function()
    starting_epoch = isempty(state) ? 0 : state[end, :epoch]
    set_or_add!(x, I...) = ismissing(state[I...]) ? state[I...] = x/length(train_data) : state[I...] += x/length(train_data)
    for epoch in starting_epoch .+ (1:settings["optimizer"]["epochs"])
        # Check for timeout
        (Dates.now() - SCRIPT_TIME_START > SCRIPT_TIMEOUT) && break

        # Training epoch
        @timeit timer "epoch" begin
            # Pre-training callbacks
            @timeit timer "pretraincbs" pretraincbs()

            # Training loop
            push!(state, [epoch; :train; missings(size(state,2)-2)])

            @timeit timer "train loop" for d in train_data
                @timeit timer "forward" ℓ, back = Zygote.pullback(() -> H_loss(d...), Flux.params(m))
                @timeit timer "reverse" gs = back(1)
                @timeit timer "update!" Flux.Optimise.update!(opt, Flux.params(m), gs)

                # Update training losses periodically
                set_or_add!(ℓ, lastindex(state,1), :loss)
                if mod(epoch, 50) == 0 #TODO FIXME
                    @timeit timer "θerr" set_or_add!(θerr(labelbatch(d)...), lastindex(state,1), :labelerr)
                    @timeit timer "θacc" set_or_add!(θacc(labelbatch(d)...), lastindex(state,1), :acc)
                    @timeit timer "ELBO" set_or_add!(L_loss(d...),  lastindex(state,1), :ELBO)
                    @timeit timer "KL"   set_or_add!(KL_loss(d...), lastindex(state,1), :KL)
                end
            end

            # Testing evaluation
            push!(state, [epoch; :test; missings(size(state,2)-2)])

            if mod(epoch, 50) == 0 #TODO FIXME
                @timeit timer "test eval" begin
                    @timeit timer "θerr" state[end, :labelerr] = θerr(labelbatch(test_data)...)
                    @timeit timer "θacc" state[end, :acc]      = θacc(labelbatch(test_data)...)
                    @timeit timer "H"    state[end, :loss]     = H_loss(test_data...)
                    @timeit timer "ELBO" state[end, :ELBO]     = L_loss(test_data...)
                    @timeit timer "KL"   state[end, :KL]       = KL_loss(test_data...)
                end
            end

            # Post-training callbacks
            @timeit timer "posttraincbs" posttraincbs()
        end

        if mod(epoch, 1000) == 0 #TODO FIXME
            show(stdout, timer); println("\n")
            show(stdout, last(state, 10)); println("\n")
        end
        (epoch == starting_epoch + 1) && TimerOutputs.reset_timer!(timer) # throw out initial loop (precompilation, first plot, etc.)
    end
end

@info("Beginning training loop...")
try
    train_loop!() #TODO FIXME
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
best_model   = SAVE ? BSON.load(savepath("log", "model-best.bson"))[:model] : deepcopy(m); #TODO FIXME
# best_model   = deepcopy(m); #TODO FIXME
eval_data    = test_data
# eval_data  = (hcat(thetas.(BT_train_data)..., thetas(BT_test_data)), cat(signals.(BT_train_data)..., signals(BT_test_data); dims = 4))
true_thetas  = thetas(eval_data) |> Flux.cpu;
true_signals = signals(eval_data) |> Flux.cpu;
model_mu_std = best_model(true_signals; nsamples = 1000, stddev = true) |> Flux.cpu;
model_thetas, model_stds = model_mu_std[1:end÷2, ..], model_mu_std[end÷2+1:end, ..];

prediction_hist = function()
    pred_hist = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        err = scale .* (model_thetas[i,:] .- true_thetas[i,:])
        s = x -> round(x; sigdigits = 2)
        mae_err, mean_err, σ, IQR = s(mean(abs, err)), s(mean(err)), s(std(err)), s(iqr(err))
        p = plot()
        histogram!(p, err;
            nbins = 50, normalized = true, grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
            title = "|μ| = $mae_err, μ = $mean_err, σ = $σ", #, IQR = $IQR",
        )
        xlims!(p, mean_err - 3σ, mean_err + 3σ)
    end
    plot([pred_hist(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting inference error histograms..."
fig = prediction_hist(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.histogram.png"))

prediction_hist_single = function()
    I = rand(1:batchsize(signals(eval_data)))
    true_thetas_sample  = thetas(eval_data)[:,I];
    true_signals_sample = repeat(signals(eval_data)[:,1,1,I], 1, 1, 1, 1000);
    model_thetas_sample = best_model(true_signals_sample; nsamples = 1, stddev = false);

    pred_hist_single = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        lab = settings["data"]["info"]["labinfer"][i]
        err = scale .* (model_thetas_sample[i,:] .- true_thetas_sample[i])
        mae_err, mean_err = round(mean(abs.(err)); sigdigits = 2), round(mean(err); sigdigits = 2)
        σ, IQR = round(std(err); sigdigits = 2), round(iqr(err); sigdigits = 2)

        p = plot()
        histogram!(p,
            scale .* model_thetas_sample[i,:];
            nbins = 20, normalized = true, grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = "$lab [$units]", title = "|μ| = $mae_err, μ = $mean_err, σ = $σ", #, IQR = $IQR",
        )
        vline!(p, scale .* [true_thetas_sample[i] mean(model_thetas_sample[i,:])]; line = ([:red :green], 4), label = ["$lab true" "$lab pred"])
    end

    ydata = true_signals_sample[:,1,1,1]
    annot = mapreduce((x, y) -> join([x,y], "\n"), settings["data"]["info"]["labinfer"], settings["data"]["info"]["labunits"], true_thetas_sample) do lab, units, θ
        "$lab = $(round(θ; sigdigits=3)) [$units]"
    end
    plot(
        plot([pred_hist_single(i) for i in 1:size(model_thetas_sample, 1)]...),
        plot(ydata; title = "MSE Signal vs. Echo Number", leg = nothing, xlims = (1,32), titlefontsize = 10, annotate = (25, 0.75*maximum(ydata), Plots.text(annot, 8))),
        layout = @layout([a{0.8h}; b{0.2h}]), # layout = @layout([a{0.8w} b{0.2w}]),
    )
end
@info "Plotting sample parameter inference histograms..."
fig = prediction_hist_single(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.histogram.single.png"))

prediction_ribbon = function()
    pred_ribbon = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]

        isort = sortperm(true_thetas[i,:])
        x = scale .* true_thetas[i, isort]
        y = scale .* model_thetas[i, isort]
        _idx = partition(1:length(x), length(x)÷20 + 1) # partition into at most 20 subgroups
        _x = [mean(x[idx]) for idx in _idx]
        _y = [mean(y[idx]) for idx in _idx]
        _σ = [std(y[idx] .- x[idx]) for idx in _idx]

        p = plot(_x, _y;
            ribbon = _σ, fillalpha = 0.5, marker = :circle, markersize = 2, grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
        )
        p = plot!(p, identity, ylims(p)...; line = (:dash, 2, :red), label = L"y = x")
        p = xlims!(p, _x[1], _x[end])
    end
    plot([pred_ribbon(i) for i in 1:size(model_thetas, 1)]...)
end
@info "Plotting prediction ribbon plots..."
fig = prediction_ribbon(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.ribbon.png"))

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
#=
@info "Plotting prediction scatter plots..."
fig = prediction_scatter(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.scatter.png"))
=#

prediction_corrplot = function()
    # cosalpha(x) = (out = copy(x); @views out[1,:] .= cosd.(out[1,:]); out)
    θ = true_thetas .* settings["data"]["info"]["labscale"] |> permutedims
    Δ = abs.(model_thetas .- true_thetas) .* settings["data"]["info"]["labscale"] |> permutedims
    # θlabs = settings["data"]["info"]["labinfer"] .* " [" .* settings["data"]["info"]["labunits"] .* "]" |> permutedims
    θlabs = settings["data"]["info"]["labinfer"] |> permutedims
    Δlabs = θlabs .* " error"
    θidx = 1:length(θlabs) # 1:2
    Δidx = 1:length(Δlabs)
    corrdata = hcat(θ[..,θidx], Δ[..,Δidx])
    corrlabs = hcat(θlabs[..,θidx], Δlabs[..,Δidx])
    fig = corrplot(corrdata; label = corrlabs, fillcolor = :purple, markercolor = cgrad(:rainbow, :misc), xrotation = 45.0, guidefontsize = 8, tickfontsize = 8)
    # savefig(fig, "tmp.pdf")
end
#=
@info "Plotting correlation plots..."
fig = prediction_corrplot(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.corrplot.png"))
=#

forward_plot = function()
    forward_rmse = function(i)
        y = sum(signals(eval_data)[:,1,:,i]; dims = 2) # Assumes signal is split linearly into channels
        z_class = MWFLearning.forward_physics_14arg(true_thetas[:,i:i]) # Forward simulation of true parameters
        z_model = MWFLearning.forward_physics_14arg(model_thetas[:,i:i]) # Forward simulation of model predicted parameters
        return (e_class = rmsd(y, z_class), e_model = rmsd(y, z_model))
    end
    errors = [forward_rmse(i) for i in 1:batchsize(thetas(eval_data))]
    e_class = (e -> e.e_class).(errors)
    e_model = (e -> e.e_model).(errors)
    p = scatter([e_class e_model];
        labels = ["RMSE: Classical" "RMSE: Model"],
        marker = [:circle :square],
        grid = true, minorgrid = true, titlefontsize = 10, ylim = (0, 0.05)
    )
end
#=
@info "Plotting forward simulation error plots..."
fig = forward_plot()
display(fig) && savefig(fig, savepath("plots", "forwarderror.png"))
=#

errorvslr = function()
    x = [lrfun.(state[:callbacks][:training][:epoch]), lrfun.(state[:callbacks][:testing][:epoch])]
    y = [state[:callbacks][:training][:loss], state[:callbacks][:testing][:loss]]
    plot(
        plot(x, (e -> log10.(e .- minimum(e) .+ 1e-6)).(y); xscale = :log10, ylabel = "stretched loss ($(settings["model"]["loss"]))", label = ["training" "testing"]),
        plot(x, (e -> log10.(e)).(y); xscale = :log10, xlabel = "learning rate", ylabel = "loss ($(settings["model"]["loss"]))", label = ["training" "testing"]);
        layout = (2,1)
    )
end
#=
@info "Plotting errors vs. learning rate..."
fig = errorvslr()
display(fig) && savefig(fig, savepath("plots", "lossvslearningrate.png"))
=#

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
#=
@info "Plotting mwf error vs. thetas..."
mwferrorvsthetas()
=#

nothing
