# ---------------------------------------------------------------------------- #
# Utils
# ---------------------------------------------------------------------------- #

function load_settings(
        default_settings_file,
    )
    # Load default settings + merge in custom settings, if given
    settings = TOML.parsefile(default_settings_file)
    mergereducer!(x, y) = deepcopy(y) # fallback
    mergereducer!(x::Dict, y::Dict) = merge!(mergereducer!, x, y)
    haskey(ENV, "SETTINGSFILE") && merge!(mergereducer!, settings, TOML.parsefile(ENV["SETTINGSFILE"]))

    # Save + print resulting settings
    outpath = settings["data"]["out"]
    !isdir(outpath) && mkpath(outpath)
    open(joinpath(outpath, "settings.toml"); write = true) do io
        TOML.print(io, settings)
    end
    TOML.print(stdout, settings)

    return settings
end

# Saving, formatting
getnow() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")
savebson(filename, data::Dict) = @elapsed BSON.bson(filename, data)
# gitdir() = realpath(DrWatson.projectdir(".."))

# Mini batching
make_minibatch(features, labels, idxs) = (features[.., idxs], labels[.., idxs])
function training_batches(features, labels, minibatchsize; overtrain = false)
    @assert batchsize(features) == batchsize(labels)
    batches = Iterators.partition(1:batchsize(features), minibatchsize)
    if overtrain
        train_set = [make_minibatch(features, labels, batches[1])]
    else
        train_set = [make_minibatch(features, labels, b) for b in batches]
    end
end
testing_batches(features, labels) = make_minibatch(features, labels, :)
features(batch) = batch[1]
labels(batch) = batch[2]

function param_summary(model, train_set, test_set)
    test_dofs = length(test_set[2])
    train_dofs = sum(batch -> length(batch[2]), train_set)
    param_dofs = sum(length, Flux.params(model))
    test_param_density = param_dofs / test_dofs
    train_param_density = param_dofs / train_dofs
    @info @sprintf(" Testing parameter density: %d/%d (%.2f %%)", param_dofs, test_dofs, 100 * test_param_density)
    @info @sprintf("Training parameter density: %d/%d (%.2f %%)", param_dofs, train_dofs, 100 * train_param_density)
end

# Losses
function make_losses(model, losstype, weights = nothing)
    l1 = weights == nothing ? @λ((x,y) -> sum(abs, model(x) .- y))  : @λ((x,y) -> sum(abs, weights .* (model(x) .- y)))
    l2 = weights == nothing ? @λ((x,y) -> sum(abs2, model(x) .- y)) : @λ((x,y) -> sum(abs2, weights .* (model(x) .- y)))
    crossent = @λ((x,y) -> Flux.crossentropy(model(x), y))
    mae = @λ((x,y) -> l1(x,y) * 1 // length(y))
    mse = @λ((x,y) -> l2(x,y) * 1 // length(y))
    rmse = @λ((x,y) -> sqrt(mse(x,y)))
    mincrossent = @λ (y) -> -sum(y .* log.(y))

    lossdict = Dict("l1" => l1, "l2" => l2, "crossent" => crossent, "mae" => mae, "mse" => mse, "rmse" => rmse, "mincrossent" => mincrossent)
    if losstype ∉ keys(lossdict)
        @warn "Unknown loss $(losstype); defaulting to mse"
        losstype = "mse"
    end

    loss = lossdict[losstype]
    accloss = losstype == "crossent" ? @λ((x,y) -> loss(x,y) - mincrossent(y)) : rmse # default
    accuracy = @λ((x,y) -> 100 - 100 * accloss(x,y))
    labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ (maximum(abs.(y); dims = 2) .- minimum(abs.(y); dims = 2))))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.((model(x) .- y) ./ y); dims = 2)))
    # labelacc = @λ((x,y) -> 100 .* vec(mean(abs.(model(x) .- y); dims = 2) ./ maximum(abs.(y); dims = 2)))
    
    return @ntuple(loss, accuracy, labelacc)
end

# Optimizer
lr(opt) = opt.eta
lr!(opt, α) = (opt.eta = α; opt.eta)
lr(opt::Flux.Optimiser) = lr(opt[1])
lr!(opt::Flux.Optimiser, α) = lr!(opt[1], α)

fixedlr(e, opt) = lr(opt) # Fixed learning rate
geometriclr(e, opt; rate = 100, factor = √10) = mod(e, rate) == 0 ? lr(opt) / factor : lr(opt) # Drop lr every `rate` epochs
clamplr(e, opt; lower = -Inf, upper = Inf) = clamp(lr(opt), lower, upper) # Clamp learning rate in [lower, upper]
findlr(e, opt; epochs = 100, minlr = 1e-6, maxlr = 0.5) = e <= epochs ? logspace(1, epochs, minlr, maxlr)(e) : maxlr # Learning rate finder
cyclelr(e, opt; lrstart = 1e-5, lrmin = 1e-6, lrmax = 1e-2, lrwidth = 50, lrtail = 5) = # Learning rate cycling
                     e <=   lrwidth          ? linspace(        1,            lrwidth, lrstart,   lrmax)(e) :
      lrwidth + 1 <= e <= 2*lrwidth          ? linspace(  lrwidth,          2*lrwidth,   lrmax, lrstart)(e) :
    2*lrwidth + 1 <= e <= 2*lrwidth + lrtail ? linspace(2*lrwidth, 2*lrwidth + lrtail, lrstart,   lrmin)(e) :
    lrmin

function make_variancelr(state, opt; rate = 250, factor = √10, stdthresh = Inf)
    last_lr_update = 0
    function lrfun(e)
        if isempty(state)
            return lr(opt)
        elseif e > 2 * rate && e - last_lr_update > rate
            min_epoch = max(1, min(state[end, :epoch] - rate, rate))
            df = dropmissing(state[(state.dataset .== :test) .& (min_epoch .<= state.epoch), [:epoch, :loss]])
            if !isempty(df) && std(df[!, :loss]) > stdthresh
                last_lr_update = e
                return lr(opt) / √10
            else
                return lr(opt)
            end
        else
            return lr(opt)
        end
    end
end

####
#### Callbacks
####

function epochthrottle(f, state, epoch_rate)
    last_epoch = 0
    function epochthrottled(args...; kwargs...)
        isempty(state) && return nothing
        epoch = state[end, :epoch]
        if epoch >= last_epoch + epoch_rate
            last_epoch = epoch
            f(args...; kwargs...)
        else
            nothing
        end
    end
end

function log10ticks(a, b; baseticks = 1:10)
    l, u = floor(Int, log10(a)), ceil(Int, log10(b))
    ticks = unique!(vcat([10.0^x .* baseticks for x in l:u]...))
    return filter!(x -> a ≤ x ≤ b, ticks)
end
function slidingindices(epoch, window = 100)
    i1_window = findfirst(e -> e ≥ epoch[end] - window + 1, epoch)
    i1_first  = findfirst(e -> e ≥ window, epoch)
    (i1_first === nothing) && (i1_first = 1)
    return min(i1_window, i1_first) : length(epoch)
end

function make_test_err_cb(state, lossfun, accfun, laberrfun, test_set)
    function()
        update_time = @elapsed begin
            if !isempty(state)
                row = findlast(==(:test), state.dataset)
                state[row, :loss]     = Flux.cpu(lossfun(test_set...))
                state[row, :acc]      = Flux.cpu(accfun(test_set...))
                state[row, :labelerr] = Flux.cpu(laberrfun(test_set...))
            end
        end
        # @info @sprintf("[%d] -> Updating testing error... (%d ms)", state[row, :epoch], 1000 * update_time)
    end
end
function make_train_err_cb(state, lossfun, accfun, laberrfun, train_set)
    function()
        update_time = @elapsed begin
            if !isempty(state)
                row = findlast(==(:train), state.dataset)
                state[row, :loss]     = mean([Flux.cpu(lossfun(b...))   for b in train_set])
                state[row, :acc]      = mean([Flux.cpu(accfun(b...))    for b in train_set])
                state[row, :labelerr] = mean([Flux.cpu(laberrfun(b...)) for b in train_set])
            end
        end
        # @info @sprintf("[%d] -> Updating training error... (%d ms)", state[row, :epoch], 1000 * update_time)
    end
end
function make_plot_errs_cb(state, filename = nothing; labelnames = "")
    function err_subplots()
        ps = Any[]
        for dataset in unique(state.dataset)
            window = 100
            min_epoch = max(1, min(state[end, :epoch] - window, window))
            df = state[(state.dataset .== dataset) .& (min_epoch .<= state.epoch), :]

            commonkw = (xscale = :log10, xticks = log10ticks(df[1, :epoch], df[end, :epoch]), xrotation = 75.0, xformatter = x->string(round(Int,x)), lw = 3, titlefontsize = 8, tickfontsize = 6, legend = :best, legendfontsize = 6)
            logspacing!(dfp) = isempty(dfp) ? dfp : unique(round.(Int, 10.0 .^ range(log10.(dfp.epoch[[1,end]])...; length = 10000))) |> I -> length(I) ≥ 5000 ? deleterows!(dfp, findall(!in(I), dfp.epoch)) : dfp

            dfp = logspacing!(dropmissing(df[!, [:epoch, :loss]]))
            p1 = plot()
            if !isempty(dfp)
                minloss = round(minimum(dfp.loss); sigdigits = 4)
                p1 = @df dfp plot(:epoch, :loss; title = "Loss ($dataset): min = $minloss)", label = "loss", ylim = (minloss, quantile(dfp.loss, 0.99)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :acc]]))
            p2 = plot()
            if !isempty(dfp)
                maxacc = round(maximum(dfp.acc); sigdigits = 4)
                p2 = @df dfp plot(:epoch, :acc; title = "Accuracy ($dataset): peak = $maxacc%)", label = "acc", yticks = 50:0.1:100, ylim = (clamp(maxacc, 50, 99) - 1.5, min(maxacc + 0.5, 100.0)), commonkw...)
            end

            dfp = logspacing!(dropmissing(df[!, [:epoch, :labelerr]]))
            p3 = plot()
            if !isempty(dfp)
                labelerr = permutedims(reduce(hcat, dfp[!, :labelerr]))
                labcol = size(labelerr,2) == 1 ? :blue : permutedims(RGB[cgrad(:darkrainbow)[z] for z in range(0.0, 1.0, length = size(labelerr,2))])
                p3 = @df dfp plot(:epoch, labelerr; title = "Label Error ($dataset): rel. %)", label = labelnames, c = labcol, yticks = 0:100, ylim = (0, min(50, maximum(labelerr[end,:]) + 3.0)), commonkw...)
            end

            push!(ps, plot(p1, p2, p3; layout = (1,3)))
        end
        plot(ps...; layout = (length(ps), 1))
    end
    function()
        try
            plot_time = @elapsed begin
                fig = err_subplots()
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_checkpoint_state_cb(state, filename = nothing; filtermissings = false, filternans = false)
    function()
        save_time = @elapsed let state = deepcopy(state)
            if !isnothing(filename)
                filtermissings && dropmissing!(state) # drop rows with missings
                filternans && filter!(r -> all(x -> !((x isa Number && isnan(x)) || (x isa AbstractArray{<:Number} && any(isnan, x))), r), state) # drop rows with NaNs
                savebson(filename, @dict(state))
            end
        end
        # @info @sprintf("[%d] -> Error checkpoint... (%d ms)", state[end, :epoch], 1000 * save_time)
    end
end
function make_plot_gan_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                fig = @df state plot(:epoch, [:Gloss :Dloss :D_x :D_G_z]; label = ["G Loss" "D loss" "D(x)" "D(G(z))"], lw = 3)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_plot_ligocvae_losses_cb(state, filename = nothing)
    function()
        try
            plot_time = @elapsed begin
                ps = Any[]
                for dataset in unique(state.dataset)
                    window = 100
                    min_epoch = max(1, min(state[end, :epoch] - window, window))
                    logspacing!(dfp) = isempty(dfp) ? dfp : unique(round.(Int, 10.0 .^ range(log10.(dfp.epoch[[1,end]])...; length = 10000))) |> I -> length(I) ≥ 5000 ? deleterows!(dfp, findall(!in(I), dfp.epoch)) : dfp

                    dfp = logspacing!(dropmissing(state[(state.dataset .== dataset) .& (min_epoch .<= state.epoch), [:epoch, :ELBO, :KL, :loss]]))
                    p = plot()
                    if !isempty(dfp)
                        commonkw = (xaxis = (:log10, log10ticks(dfp[1, :epoch], dfp[end, :epoch])), xrotation = 60.0, legend = :best, lw = 3, xformatter = x->string(round(Int,x)))
                        pKL   = @df dfp plot(:epoch, :KL;   title =   "KL vs. epoch ($dataset): max = $(round(maximum(dfp.KL);   sigdigits = 4))", lab = "KL",   c = :orange, commonkw...)
                        pELBO = @df dfp plot(:epoch, :ELBO; title = "ELBO vs. epoch ($dataset): min = $(round(minimum(dfp.ELBO); sigdigits = 4))", lab = "ELBO", c = :blue,   commonkw...)
                        pH    = @df dfp plot(:epoch, :loss; title =    "H vs. epoch ($dataset): min = $(round(minimum(dfp.loss); sigdigits = 4))", lab = "loss", c = :green,  commonkw...)
                        p     = plot(pKL, pELBO, pH; layout = (3,1))
                    end
                    push!(ps, p)
                end
                fig = plot(ps...)
                !(filename === nothing) && savefig(fig, filename)
                display(fig)
            end
            # @info @sprintf("[%d] -> Plotting progress... (%d ms)", state[end, :epoch], 1000 * plot_time)
        catch e
            if e isa InterruptException
                rethrow(e) # Training interrupted by user
            else
                @info @sprintf("[%d] -> PLOTTING FAILED...", state[end, :epoch])
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
    end
end
function make_update_lr_cb(state, opt, lrfun; lrcutoff = 1e-6)
    last_lr = nothing
    curr_lr = lr(opt)
    function()
        # Update learning rate and exit if it has become to small
        if !isempty(state)
            curr_lr = lr!(opt, lrfun(state[end, :epoch]))
        end
        if curr_lr < lrcutoff
            @info(" -> Early-exiting: Learning rate has dropped below cutoff = $lrcutoff")
            Flux.stop()
        end
        if last_lr === nothing
            @info(" -> Initial learning rate: " * @sprintf("%.2e", curr_lr))
        elseif last_lr != curr_lr
            @info(" -> Learning rate updated: " * @sprintf("%.2e", last_lr) * " --> "  * @sprintf("%.2e", curr_lr))
        end
        last_lr = curr_lr
        return nothing
    end
end
function make_save_best_model_cb(state, model, opt, filename = nothing)
    function()
        # If this is the best accuracy we've seen so far, save the model out
        isempty(state) && return nothing
        df = state[state.dataset .== :test, :]
        ismissing(df.acc[end]) && return nothing
        isempty(skipmissing(df.acc)) && return nothing

        best_acc = maximum(skipmissing(df.acc))
        if df[end, :acc] == best_acc
            try
                save_time = @elapsed let model = Flux.cpu(deepcopy(model)) #, opt = Flux.cpu(deepcopy(opt))
                    # weights = collect(Flux.params(model))
                    # !(filename === nothing) && savebson(filename * "weights-best.bson", @dict(weights))
                    !(filename === nothing) && savebson(filename * "model-best.bson", @dict(model))
                    # !(filename === nothing) && savebson(filename * "opt-best.bson", @dict(opt)) #TODO BSON optimizer saving broken
                end
                @info @sprintf("[%d] -> New best accuracy %.4f; model saved (%4d ms)", df[end, :epoch], df[end, :acc], 1000 * save_time)
            catch e
                @warn "Error saving best model..."
                @warn sprint(showerror, e, catch_backtrace())
            end
        end
        nothing
    end
end
function make_checkpoint_model_cb(state, model, opt, filename = nothing)
    function()
        try
            save_time = @elapsed let model = Flux.cpu(deepcopy(model)) #, opt = Flux.cpu(deepcopy(opt))
                # weights = collect(Flux.params(model))
                # !(filename === nothing) && savebson(filename * "weights-checkpoint.bson", @dict(weights))
                !(filename === nothing) && savebson(filename * "model-checkpoint.bson", @dict(model))
                # !(filename === nothing) && savebson(filename * "opt-checkpoint.bson", @dict(opt)) #TODO BSON optimizer saving broken
            end
            # @info @sprintf("[%d] -> Model checkpoint... (%d ms)", state[end, :epoch], 1000 * save_time)
        catch e
            @warn "Error checkpointing model..."
            @warn sprint(showerror, e, catch_backtrace())
        end
    end
end
