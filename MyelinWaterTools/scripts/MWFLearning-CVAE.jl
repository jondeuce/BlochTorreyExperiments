# Initialization project/code loading
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.instantiate()

using Statistics: mean, median, std
using StatsBase: quantile, sample, iqr
using Base.Iterators: repeated, partition
using Printf

using MWFLearning
# using CuArrays
pyplot(size=(800,600))
# pyplot(size=(1600,900))

# Settings
const default_settings_file = "/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MyelinWaterTools/MWFLearning/examples/cvae_settings.toml"
const settings = let
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(default_settings_file)
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
    if haskey(ENV, "SETTINGSFILE")
        merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))
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
function x_sampler()
    out = zeros(T, 15, MB_train_batch_size)
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
        K      = 1e-3 # Fix mock signals at constant near-zero permeability (lower bound of MWFLearning.log10sampler(1e-3, 10.0))
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
        # out[1,j]  = log10(TE*K) # log(TE*Kperm)
        # out[2,j]  = cosd(alpha) # cosd(alpha)
        # out[3,j]  = g # gratio
        # out[4,j]  = mwf # mwf
        # out[5,j]  = T2sp / TE # T2mw/TE
        # out[6,j]  = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        # out[7,j]  = iwf # iwf
        # out[8,j]  = ewf # ewf
        # out[9,j]  = iwf + ewf # iewf
        # out[10,j] = T2lp / TE # T2iw/TE
        # out[11,j] = T2tiss / TE # T2ew/TE
        # out[12,j] = T1sp / TE # T1mw/TE
        # out[13,j] = T1lp / TE # T1iw/TE
        # out[14,j] = T1tiss / TE # T1ew/TE
        # out[15,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
        # out[1,j]  = cosd(alpha) # cosd(alpha)
        # out[2,j]  = g # gratio
        # out[3,j]  = mwf # mwf
        # out[4,j]  = T2sp / TE # T2mw/TE
        # out[5,j]  = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        # out[6,j]  = log10(TE*K) # log(TE*Kperm)
        # out[7,j]  = iwf # iwf
        # out[8,j]  = ewf # ewf
        # out[9,j]  = iwf + ewf # iewf
        # out[10,j] = T2lp / TE # T2iw/TE
        # out[11,j] = T2tiss / TE # T2ew/TE
        # out[12,j] = T1sp / TE # T1mw/TE
        # out[13,j] = T1lp / TE # T1iw/TE
        # out[14,j] = T1tiss / TE # T1ew/TE
        # out[15,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
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
        # out[1,j] = mwf
        # out[2,j] = 1 - mwf # iewf
        # out[3,j] = inv((ivf * inv(T2lp) + evf * inv(T2tiss)) / (ivf + evf)) / TE # T2iew/TE
        # out[4,j] = T2sp / TE # T2mw/TE
        # out[5,j] = alpha
        # out[6,j] = log10(TE*K)
        # out[7,j] = inv((ivf * inv(T1lp) + evf * inv(T1tiss)) / (ivf + evf)) / TE # T1iew/TE
        # out[8,j] = T1sp / TE # T1mw/TE
    end
    return out
end
# y_sampler(x) = (y = MWFLearning.forward_physics_8arg(x); reshape(y, size(y,1), 1, 1, :))
y_sampler(x) = (y = MWFLearning.forward_physics_14arg(x); reshape(y, size(y,1), 1, 1, :))
# y_sampler(x) = (y = MWFLearning.forward_physics_15arg(x); reshape(y, size(y,1), 1, 1, :))
# y_sampler(x) = (y = MWFLearning.forward_physics_15arg_Kperm(x); reshape(y, size(y,1), 1, 1, :))
MB_sampler = MWFLearning.LazyMiniBatches(MB_num_train_batches, x_sampler, y_sampler)
=#

# Train using Bloch-Torrey training/testing data, or sampler data
train_data, test_data = BT_train_data, BT_test_data
# train_data, test_data = MB_sampler, (x -> x[.., 1:MB_test_batch_size]).(rand(MB_sampler))

# Construct model
@info "Constructing model..."
@unpack m = MWFLearning.make_model(settings)[1] |> maybegpu;
model_summary(m, savepath("models", "architecture.txt"));
param_summary(m, labelbatch.(train_data), labelbatch(test_data));

# Loss and accuracy function
theta_weights()::CVT = inv.(settings["data"]["info"]["labwidth"][thetas_infer_idx]) .* unitsum(settings["data"]["info"]["labweights"]) |> copy |> VT |> maybegpu |> CVT
datanoise = let snr = T(settings["data"]["postprocess"]["SNR"]); snr ≤ 0 ? identity : @λ(y -> MWFLearning.add_rician(y, snr)); end
trainloss = @λ (x,y) -> MWFLearning.H_LIGOCVAE(m, x, datanoise(y))
θloss, θacc, θerr = make_losses(@λ(y -> m(datanoise(y); nsamples = 10)), settings["model"]["loss"], theta_weights())

# Optimizer
opt = Flux.ADAM(settings["optimizer"]["ADAM"]["lr"], (settings["optimizer"]["ADAM"]["beta"]...,))
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

update_lr_cb            = MWFLearning.make_update_lr_cb(state, opt, lrfun)
checkpoint_state_cb     = MWFLearning.make_checkpoint_state_cb(state, savepath("log", "errors.bson"))
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
                @timeit timer "forward" ℓ, back = Flux.Zygote.pullback(() -> trainloss(d...), Flux.params(m))
                @timeit timer "reverse" gs = back(1)
                @timeit timer "update!" Flux.Optimise.update!(opt, Flux.params(m), gs)

                # Update training losses periodically
                set_or_add!(ℓ, lastindex(state,1), :loss)
                if mod(epoch, 10) == 0 #TODO
                    @timeit timer "θerr" set_or_add!(θerr(labelbatch(d)...), lastindex(state,1), :labelerr)
                    @timeit timer "θacc" set_or_add!(θacc(labelbatch(d)...), lastindex(state,1), :acc)
                    @timeit timer "ELBO" set_or_add!(MWFLearning.L_LIGOCVAE(m, d...), lastindex(state,1), :ELBO)
                    @timeit timer "KL"   set_or_add!(MWFLearning.KL_LIGOCVAE(m, d...), lastindex(state,1), :KL)
                end
            end

            # Testing evaluation
            push!(state, [epoch; :test; missings(size(state,2)-2)])

            if mod(epoch, 10) == 0 #TODO
                @timeit timer "test eval" begin
                    @timeit timer "θerr" state[end, :labelerr] = θerr(labelbatch(test_data)...)
                    @timeit timer "θacc" state[end, :acc]      = θacc(labelbatch(test_data)...)
                    @timeit timer "H"    state[end, :loss]     = MWFLearning.H_LIGOCVAE(m, test_data...)
                    @timeit timer "ELBO" state[end, :ELBO]     = MWFLearning.L_LIGOCVAE(m, test_data...)
                    @timeit timer "KL"   state[end, :KL]       = MWFLearning.KL_LIGOCVAE(m, test_data...)
                end
            end

            # Post-training callbacks
            @timeit timer "posttraincbs" posttraincbs()
        end

        if mod(epoch, 500) == 0 #TODO
            show(stdout, timer); println("\n")
            show(stdout, last(state, 10)); println("\n")
        end
        (epoch == starting_epoch + 1) && TimerOutputs.reset_timer!(timer) # throw out initial loop (precompilation, first plot, etc.)
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
eval_data    = test_data
# eval_data  = (hcat(thetas.(BT_train_data)..., thetas(BT_test_data)), cat(signals.(BT_train_data)..., signals(BT_test_data); dims = 4))
true_thetas  = thetas(eval_data) |> Flux.cpu;
true_signals = signals(eval_data) |> Flux.cpu;
model_mu_std = best_model(true_signals; nsamples = 1000, stddev = true) |> Flux.cpu;
model_thetas, model_stds = model_mu_std[1:end÷2, ..], model_mu_std[end÷2+1:end, ..];

# keep_indices = true_thetas[2,:] .≤ -3.5 # Keep small permeability only
# true_thetas  = true_thetas[.., keep_indices]
# true_signals = true_signals[.., keep_indices]
# model_thetas = model_thetas[.., keep_indices]
# model_stds   = model_stds[.., keep_indices]

true_thetas = thetas(eval_data) #repeat(thetas(eval_data)[:,2:2], 1,500);
true_signals = signals(eval_data) #repeat(signals(eval_data)[:,:,:,2:2], 1,1,1,500);
model_thetas = best_model(true_signals; nsamples = 1, stddev = false) |> Flux.cpu;

prediction_hist = function()
    pred_hist = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        err = scale .* (model_thetas[i,:] .- true_thetas[i,:])
        abs_μ, μ = round(mean(abs.(err)); sigdigits = 2), round(mean(err); sigdigits = 2)
        σ, IQR = round(std(err); sigdigits = 2), round(iqr(err); sigdigits = 2)
        p = histogram(err; nbins = 50, normalized = true,
            grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
            title = "|μ| = $abs_μ, μ = $μ, σ = $σ", #, IQR = $IQR",
        )
        p = xlims!(p, μ - 3σ, μ + 3σ)
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

prediction_ribbon = function()
    pred_ribbon = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]

        isort = sortperm(true_thetas[i,:]);
        x = scale .* true_thetas[i, isort];
        y = scale .* model_thetas[i, isort];
        _idx = partition(1:length(x), length(x)÷20)
        _x = [mean(x[idx]) for idx in _idx]; #_x = [x[1]; _x; x[end]]
        _y = [mean(y[idx]) for idx in _idx]; #_y = [y[1]; _y; y[end]]
        _σ = [std(y[idx] .- x[idx]) for idx in _idx]; #_σ = [0; _σ; 0]

        p = plot(_x, _y; ribbon = _σ,
            fillalpha = 0.5,
            marker = :circle, markersize = 2, grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
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

prediction_hist_single = function()
    pred_hist_single = function(i)
        scale = settings["data"]["info"]["labscale"][i]
        units = settings["data"]["info"]["labunits"][i]
        err = scale .* (model_thetas[i,:] .- true_thetas[i,:])
        abs_μ, μ = round(mean(abs.(err)); sigdigits = 2), round(mean(err); sigdigits = 2)
        σ, IQR = round(std(err); sigdigits = 2), round(iqr(err); sigdigits = 2)
        
        p = histogram(scale .* model_thetas[i,:]; nbins = 20, normalized = true,
            grid = true, minorgrid = true, titlefontsize = 10, legend = :best,
            label = settings["data"]["info"]["labinfer"][i] * " [$units]",
            title = "|μ| = $abs_μ, μ = $μ, σ = $σ", #, IQR = $IQR",
        )
    end
    
    annot = [settings["data"]["info"]["labinfer"][i] * " = $(round(true_thetas[i,1];sigdigits=3)) [" * settings["data"]["info"]["labunits"][i] * "]" for i in 1:5] |> t -> join(t, "\n")
    # annot = [L"\cos\alpha = -0.95", L"g = 0.74", L"MWF = 0.23", L"T_2^{MW}/TE = 3.0", L"T_2^{IEW}/TE = 10.0"] |> t -> join(t, "\n")
    plot(
        [pred_hist_single(i) for i in 1:size(model_thetas, 1)]...,
        plot(true_signals[:,1,1,1]; title = "MSE Signal vs. Echo Number", leg = nothing, xlims = (1,32), titlefontsize = 10, annotate = (22, 0.5, Plots.text(annot, 8))),
    )
end
@info "Plotting prediction histograms..."
fig = prediction_hist_single(); display(fig)
SAVE && savefig(fig, savepath("plots", "theta.histogram.single.png"))

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
