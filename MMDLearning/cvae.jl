# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
using MWFLearning
pyplot(size=(800,600))
Random.seed!(0);

# CVAE Settings
findendswith(dir, suffix) = filter!(s -> endswith(s, suffix), readdir(dir)) |> x -> isempty(x) ? nothing : joinpath(dir, first(x))
const default_settings_file = joinpath(@__DIR__, "src", "cvae_settings.toml")
const settings = let
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(default_settings_file)
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
    haskey(ENV, "SETTINGSFILE") && merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))
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

const savefoldernames = ["settings", "models", "log", "plots"]
const savefolders = Dict{String,String}(savefoldernames .=> mkpath.(joinpath.(settings["dir"], savefoldernames)))
savepath(folder, filename) = SAVE ? joinpath(savefolders[folder], FILE_PREFIX * filename) : nothing
if SAVE
    # Save + print resulting settings
    open(savepath("settings", "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
end

# MMD settings
const IS_TOY_MODEL = false #TODO
const CORRECT_TEST_DATA = false #TODO
const CORRECT_TRAIN_DATA = false #TODO
const mmd_settings = load_settings()
const mmd_models = let
    wrap_nottrainable(m::Dict) = Dict{String,Any}([k => NotTrainable(v) for (k,v) in m])
    load_nottrainable(s::String) = wrap_nottrainable(deepcopy(BSON.load(s)))
    IS_TOY_MODEL ?
        load_nottrainable("/project/st-arausch-1/jcd1994/MMD-Learning/toymmdopt_eps=1e-2/toymmdopt-v7/sweep/2/best-model.bson") :
        load_nottrainable("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/mmdopt-v3/sweep/92/best-model.bson")
end

# Convenience functions
split_correction_and_noise(μlogσ) = μlogσ[1:end÷2, :], exp.(μlogσ[end÷2+1:end, :])
noise_instance(X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
get_correction_and_noise(X) = split_correction_and_noise(mmd_models["mmd"](mmd_models["vae.E"](X)))
get_correction(X) = get_correction_and_noise(X)[1]
get_noise(X) = get_correction_and_noise(X)[2]
get_noise_instance(X) = noise_instance(X, get_noise(X))
get_corrected_signal(X) = get_corrected_signal(X, get_correction_and_noise(X)...)
get_corrected_signal(X, dX, ϵ) = get_corrected_signal(abs.(X .+ dX), ϵ)
function get_corrected_signal(X, ϵ)
    ϵR, ϵI = noise_instance(X, ϵ), noise_instance(X, ϵ)
    Xϵ = @. sqrt((X + ϵR)^2 + ϵI^2)
    # !IS_TOY_MODEL && (Xϵ = Xϵ ./ sum(Xϵ; dims = 1)) #TODO don't need to normalize learned corrections
    return Xϵ
end

# Load and prepare signal data
@info "Preparing data..."
function make_samples(n; dataset, correction)
    θ, Y = if IS_TOY_MODEL
        # train data has different distribution (exponent 4.0), test data has "true" distribution (exponent 2.0)
        power = dataset === :train ? T(4.0) : T(2.0)
        θ = toy_theta_sampler(n)
        Y = toy_signal_model(θ, nothing, power)
        θ, Y
    else
        # train data has different distribution (EPG model), test data has "true" distribution (MMD-corrected EPG model)
        sampleY, _, sampleθ = make_mle_data_samplers(
            mmd_settings["prior"]["data"]["image"]::String,
            mmd_settings["prior"]["data"]["thetas"]::String;
            ntheta = mmd_settings["data"]["ntheta"]::Int,
            padtrain = settings["data"]["padtrain"]::Bool,
            filteroutliers = true,
            plothist = false, #TODO
        )
        θ = copy(sampleθ(nothing; dataset = dataset)) #TODO
        Y = copy(sampleY(nothing; dataset = dataset)) #TODO

        if settings["data"]["info"]["labnames"][1] == "cosd(alpha)"
        θ[1,:] .= cosd.(θ[1,:]) # transform flipangle -> cosd(flipangle)
        end

        if settings["data"]["info"]["labnames"][3] == "T2long"
        θ[3,:] .= θ[2,:] .+ θ[3,:] # transform dT2 -> T2long = T2short + dT2
        end

        for i in 1:size(θ,1)
            lo, hi = extrema(θ[i,:])
            settings["data"]["info"]["labmean"][i] = (hi + lo)/2
            settings["data"]["info"]["labwidth"][i] = hi - lo
        end

        θ, Y
    end

    # apply learned correction
    if correction #TODO
        Y .= abs.(Y .+ get_correction(Y))
end

    Y = reshape(Y, (size(Y,1), 1, 1, size(Y,2)))
    return θ, Y
end
const train_data = training_batches(make_samples(settings["data"]["ntrain"]; dataset = :train, correction = CORRECT_TRAIN_DATA)..., settings["data"]["train_batch"])
const test_data = testing_batches(make_samples(settings["data"]["ntest"]; dataset = :test, correction = CORRECT_TEST_DATA)...)

thetas = batch -> features(batch)
signals = batch -> labels(batch)
labelbatch(batch) = (signals(batch), thetas(batch))

# Construct model
@info "Constructing model..."
m = MWFLearning.make_model(settings, "DenseLIGOCVAE");
model_summary(m, savepath("models", "architecture.txt"));
param_summary(m, labelbatch.(train_data), labelbatch(test_data));

# Loss and accuracy function
theta_weights()::VT = inv.(settings["data"]["info"]["labwidth"]) .* unitsum(settings["data"]["info"]["labweights"]) |> copy |> VT
function rician_data_noise(y)
    noise = T(settings["data"]["postprocess"]["noise"]::Float64)
    y = noise > 0 ? rand.(Rician.(y, noise)) : y
    nrm = settings["data"]["preprocess"]["normalize"]::String
    y = nrm == "unitsum" ? unitsum(y; dims = 1) : y
    return y
end
function learned_data_noise(y)
    _y = DenseResize()(y)
    y = reshape(get_corrected_signal(_y, get_noise(_y)), size(y))
    # nrm = settings["data"]["preprocess"]["normalize"]::String
    # y = nrm == "unitsum" ? unitsum(y; dims = 1) : y #TODO don't need to normalize learned corrections
    return y
end
test_data_noise(y) = CORRECT_TEST_DATA ? learned_data_noise(y) : rician_data_noise(y)
train_data_noise(y) = CORRECT_TRAIN_DATA ? learned_data_noise(y) : rician_data_noise(y)
test_data_noise(d::Tuple) = (d[1], test_data_noise(d[2]))
train_data_noise(d::Tuple) = (d[1], train_data_noise(d[2]))

#= #TODO
plot(
    plot(((test_data[2])[:,1,1,1:3]); title = "test"),
    plot(test_data_noise((test_data[2])[:,1,1,1:3]); title = "noisy test"),
    plot(((train_data[1][2])[:,1,1,4:6]); title = "train"),
    plot(train_data_noise((train_data[1][2])[:,1,1,4:6]); title = "noisy train"),
) |> display
=#

H_loss  = @λ (x,y) -> MWFLearning.H_LIGOCVAE(m, x, y; gamma = T(settings["model"]["gamma"]))
L_loss  = @λ (x,y) -> MWFLearning.L_LIGOCVAE(m, x, y)
KL_loss = @λ (x,y) -> MWFLearning.KL_LIGOCVAE(m, x, y)
θloss, θacc, θerr = make_losses(@λ(y -> m(y; nsamples = 10)), settings["model"]["loss"], theta_weights())

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
blanktrainrow = push!(deepcopy(state), [0; :train; missings(size(state,2)-2)])
blanktestrow  = push!(deepcopy(state), [0; :test; missings(size(state,2)-2)])

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
    Flux.throttle(checkpoint_state_cb, 300; leading = false), #TODO FIXME (300)
    Flux.throttle(checkpoint_model_cb, 300; leading = false), #TODO FIXME (300)
    Flux.throttle(plot_errs_cb, 300; leading = false), #TODO FIXME (300)
    Flux.throttle(plot_ligocvae_losses_cb, 300; leading = false), #TODO FIXME (300)
])

# Training Loop
train_loop! = function()
    starting_epoch = isempty(state) ? 0 : state[end, :epoch]
    set_or_add!(r, x, col) = ismissing(r[end,col]) ? r[end,col] = x/length(train_data) : r[end,col] += x/length(train_data)
    for epoch in starting_epoch .+ (1:settings["optimizer"]["epochs"])
        # Check for timeout
        (Dates.now() - SCRIPT_TIME_START > SCRIPT_TIMEOUT) && break

        # Training epoch
        @timeit timer "epoch" begin
            # Pre-training callbacks
            @timeit timer "pretraincbs" pretraincbs()

            # Training loop
            newtrainrow = deepcopy(blanktrainrow)
            newtrainrow[end, :epoch] = epoch
            @timeit timer "train loop" for d in train_data
                d = train_data_noise(d) # add unique noise instance
                @timeit timer "forward" ℓ, back = Zygote.pullback(() -> H_loss(d...), Flux.params(m))
                @timeit timer "reverse" gs = back(1)
                @timeit timer "update!" Flux.Optimise.update!(opt, Flux.params(m), gs)

                # Update training losses periodically
                set_or_add!(newtrainrow, ℓ, :loss)
                if mod(epoch, 10) == 0 #TODO FIXME (10)
                    @timeit timer "θerr" set_or_add!(newtrainrow, θerr(labelbatch(d)...), :labelerr)
                    @timeit timer "θacc" set_or_add!(newtrainrow, θacc(labelbatch(d)...), :acc)
                    @timeit timer "ELBO" set_or_add!(newtrainrow, L_loss(d...),  :ELBO)
                    @timeit timer "KL"   set_or_add!(newtrainrow, KL_loss(d...), :KL)
                end
            end
            append!(state, newtrainrow)

            # Testing evaluation
            newtestrow = deepcopy(blanktestrow)
            newtestrow[end, :epoch] = epoch
            if mod(epoch, 10) == 0 #TODO FIXME (10)
                @timeit timer "test eval" begin
                    d = test_data_noise(test_data) # add unique noise instance
                    @timeit timer "θerr" newtestrow[end, :labelerr] = θerr(labelbatch(d)...)
                    @timeit timer "θacc" newtestrow[end, :acc]      = θacc(labelbatch(d)...)
                    @timeit timer "H"    newtestrow[end, :loss]     = H_loss(d...)
                    @timeit timer "ELBO" newtestrow[end, :ELBO]     = L_loss(d...)
                    @timeit timer "KL"   newtestrow[end, :KL]       = KL_loss(d...)
                end
            end
            append!(state, newtestrow)

            # Post-training callbacks
            @timeit timer "posttraincbs" posttraincbs()
        end

        if mod(epoch, 100) == 0 #TODO FIXME (100)
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
best_model   = SAVE ? BSON.load(savepath("log", "model-best.bson"))[:model] : deepcopy(m); #TODO
# best_model   = deepcopy(m); #TODO FIXME
eval_data    = test_data; #TODO FIXME
true_thetas  = thetas(eval_data);
true_signals = signals(eval_data);
# eval_data    = test_data; #TODO FIXME
# true_thetas  = thetas(eval_data)[:,1:min(10000,end)];
# true_signals = signals(eval_data)[:,:,:,1:min(10000,end)];
# eval_data    = (reduce(hcat, (x->x[1]).(train_data)), reduce((x,y) -> cat(x,y;dims=4), (x->x[2]).(train_data))); #TODO FIXME
# true_thetas  = thetas(eval_data)[:,1:min(10000,end)];
# true_signals = signals(eval_data)[:,:,:,1:min(10000,end)];
model_mu_std = best_model(test_data_noise(true_signals); nsamples = 1000, stddev = true); #TODO
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
    I = rand(1:batchsize(true_signals))
    true_thetas_sample  = true_thetas[:,I];
    true_signals_sample = repeat(true_signals[:,1,1,I], 1, 1, 1, 1000);
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

nothing
