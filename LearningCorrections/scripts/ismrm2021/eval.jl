####
#### MLE inference
####

function mle_mri_model(
        phys::BiexpEPGModel,
        models,
        derived;
        data_source   = :image, # One of :image, :simulated
        data_subset   = :mask,  # One of :mask, :val, :train, :test
        batch_size    = 128 * Threads.nthreads(),
        initial_iter  = 10,
        verbose       = true,
        checkpoint    = false,
        dryrun        = false,
        dryrunsamples = dryrun ? batch_size : nothing,
        opt_alg       = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args      = Dict{Symbol,Any}(),
    )
    @assert data_source ∈ (:image, :simulated)
    @assert data_subset ∈ (:mask, :val, :train, :test)

    # MLE for whole image of simulated data
    Y, Yc = let
        image_ind  = phys.images[1].indices[data_subset]
        image_data = phys.images[1].data[image_ind,:] |> to32 |> permutedims
        if data_source === :image
            !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-data-$(getnow()).mat", Dict{String,Any}("Y" => arr64(image_data)))
            arr64(image_data), image_data
        else # data_source === :simulated
            X, θ, Z = sampleXθZ(derived["cvae"], derived["prior"], image_data; posterior_θ = true, posterior_Z = true)
            X̂       = sampleX̂(derived["ricegen"], X, Z)
            !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-data-$(getnow()).mat", Dict{String,Any}("X" => arr64(X), "theta" => arr64(θ), "Z" => arr64(Z), "Xhat" => arr64(X̂), "Y" => arr64(image_data)))
            arr64(X̂), X̂
        end
    end
    !isnothing(dryrunsamples) && (I = sample(MersenneTwister(0), 1:size(Y,2), dryrunsamples; replace = dryrunsamples > size(Y,2)); Y = Y[:,I]; Yc = Yc[:,I])

    initial_guess = posterior_state(
        derived["cvae"],
        derived["prior"],
        Yc;
        miniter = 1,
        maxiter = initial_iter,
        alpha   = 0.0,
        verbose = verbose,
        mode    = :maxlikelihood,
    )
    initial_guess = map(arr64, initial_guess)

    logϵlo, logϵhi = log.(extrema(vec(initial_guess.ϵ))) .|> Float64
    initial_logϵ   = log.(vec(mean(initial_guess.ϵ; dims = 1))) |> arr64
    lower_bounds   = [θlower(phys); logϵlo] |> arr64
    upper_bounds   = [θupper(phys); logϵhi] |> arr64

    #= Test random initial guess
    if initial_iter == 0
        initial_guess.θ .= sampleθprior(phys, size(Y,2)) |> arr64
        initial_logϵ    .= logϵlo .+ (logϵhi - logϵlo) .* rand(size(Y,2)) |> arr64
    end
    =#

    work_spaces = map(1:Threads.nthreads()) do _
        (
            Y   = zeros(nsignal(phys)),
            x0  = zeros(ntheta(phys) + 1),
            epg = BiexpEPGModelWork(phys),
            opt = let
                opt = NLopt.Opt(opt_alg, ntheta(phys) + 1)
                opt.lower_bounds  = lower_bounds
                opt.upper_bounds  = upper_bounds
                opt.xtol_rel      = 1e-8
                opt.ftol_rel      = 1e-8
                opt.maxeval       = 250
                opt.maxtime       = 1.0
                for (k,v) in opt_args
                    setproperty!(opt, k, v)
                end
                opt
            end,
        )
    end

    results = (
        theta     = fill(NaN, ntheta(phys), size(Y,2)),
        epsilon   = fill(NaN, size(Y,2)),
        loss      = fill(NaN, size(Y,2)),
        retcode   = fill(NaN, size(Y,2)),
        numevals  = fill(NaN, size(Y,2)),
        solvetime = fill(NaN, size(Y,2)),
    )

    function f(work, x::Vector{Float64})
        @inbounds begin
            nθ = ntheta(phys)
            θ  = ntuple(i -> x[i], nθ)
            ϵ  = exp(x[nθ+1])
            ψ  = θmodel(phys, θ...)
            X  = _signal_model_f64(phys, work.epg, ψ)
            μX = -Inf
            for i in eachindex(X)
                μX = max(μX, _rician_mean_cuda(X[i], ϵ)) # Signal normalization factor
            end
            ℓ  = 0.0
            ϵ  = ϵ / μX # Hoist outside loop
            for i in eachindex(X)
                ℓ -= _rician_logpdf_cuda(work.Y[i], X[i] / μX, ϵ) # Rician negative log likelihood
            end
            return ℓ
        end
    end

    function fg!(work, x::Vector{Float64}, g::Vector{Float64})
        if length(g) > 0
            simple_fd_gradient!(g, y -> f(work, y), x, lower_bounds, upper_bounds)
        else
            f(work, x)
        end
    end

    #= Benchmarking
    work_spaces[1].Y .= rand(nsignal(phys))
    @info "Timing function..."; @btime( $f( $( work_spaces[1] ), $( (lower_bounds .+ upper_bounds) ./ 2 ) ) ) |> display
    @info "Timing gradient..."; @btime( $fg!( $( work_spaces[1] ), $( (lower_bounds .+ upper_bounds) ./ 2 ), $( zeros(ntheta(phys) + 1) ) ) ) |> display
    =#

    start_time = time()
    batches = Iterators.partition(1:size(Y,2), batch_size)
    LinearAlgebra.BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    @time for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Y                .= Y[:,j]
                work.x0[1:end-1]      .= initial_guess.θ[:,j]
                work.x0[end]           = initial_logϵ[j]
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.theta[:,j]    .= minx[1:end-1]
                results.epsilon[j]     = minx[end]
                results.loss[j]        = minf
                results.retcode[j]     = Base.eval(NLopt, ret) |> Int #TODO cleaner way to convert Symbol to enum?
                results.numevals[j]    = work.opt.numevals
                results.solvetime[j]   = solvetime
            end
        end

        # Checkpoint results
        elapsed_time   = time() - start_time
        remaining_time = (elapsed_time / batchnum) * (length(batches) - batchnum)
        mle_per_second = batch[end] / elapsed_time
        savetime       = @elapsed !dryrun && checkpoint && DECAES.MAT.matwrite("mle-$data_source-$data_subset-results-checkpoint-$(getnow()).mat", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # checkpoint progress
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- save: $(round(savetime; digits = 2))s" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))"
    end

    !dryrun && DECAES.MAT.matwrite("mle-$data_source-$data_subset-results-final-$(getnow()).mat", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # save final results
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end

####
#### Model evaluation
####

function eval_mri_model(
        phys::BiexpEPGModel,
        models,
        derived;
        zslices = 24:24,
        naverage = 10,
        savefolder = ".",
        savetypes = [".png"],
        mle_image_path = ".",
        mle_sim_path = ".",
        force_decaes = false,
        force_histograms = false,
        posterior_mode = :maxlikelihood,
        quiet = false,
        dataset = :val, # :val or (for final model comparison) :test
    )

    inverter(Ysamples; kwargs...) = posterior_state(derived["cvae"], derived["prior"], Ysamples; verbose = !quiet, alpha = 0.0, miniter = 1, maxiter = naverage, mode = posterior_mode, kwargs...)
    saveplot(p, name, folder = savefolder) = map(suf -> savefig(p, joinpath(mkpath(folder), name * suf)), savetypes)

    flat_test(x) = flat_indices(x, phys.images[1].indices[dataset])
    flat_train(x) = flat_indices(x, phys.images[1].indices[:train])
    flat_indices(x, indices) =
        x isa AbstractMatrix ? (@assert(size(x,2) == length(indices)); return x) : # matrix with length(indices) columns
        x isa AbstractTensor4D ?
            (size(x)[1:3] == (length(indices), 1, 1)) ? permutedims(reshape(x, :, size(x,4))) : # flattened 4D array with first three dimensions (length(indices), 1, 1)
            (size(x)[1:3] == size(phys.images[1].data)[1:3]) ? permutedims(x[indices,:]) : # 4D array with first three dimensions equal to image size
            error("4D array has wrong shape") :
        error("x must be an $AbstractMatrix or an $AbstractTensor4D")

    flat_image_to_flat_test(x) = flat_image_to_flat_indices(x, phys.images[1].indices[dataset])
    flat_image_to_flat_train(x) = flat_image_to_flat_indices(x, phys.images[1].indices[:train])
    function flat_image_to_flat_indices(x, indices)
        _x = similar(x, size(x,1), size(phys.images[1].data)[1:3]...)
        _x[:, phys.images[1].indices[:mask]] = x
        return _x[:, indices]
    end

    # Compute decaes on the image data if necessary
    if !haskey(phys.meta, :decaes)
        @info "Recomputing T2 distribution for image data..."
        @time t2_distributions!(phys)
    end

    mle_image_state = let
        mle_image_results = Glob.readdir(Glob.glob"mle-image-mask-results-final-*.mat", mle_image_path) |> last |> DECAES.MAT.matread
        θ = mle_image_results["theta"] |> to32
        ϵ = reshape(exp.(mle_image_results["epsilon"] |> to32), 1, :) #TODO: should save as "logepsilon"
        ℓ = reshape(mle_image_results["loss"] |> to32, 1, :) # negative log-likelihood loss
        X = signal_model(phys, θ)
        ν, δ, Z = X, nothing, nothing
        Y = add_noise_instance(phys, X, ϵ)
        (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
    end

    let
        Y_test = phys.Y[dataset] |> to32
        Y_train = phys.Y[:train] |> to32
        Y_train_edges = Dict([k => v.edges[1] for (k,v) in phys.meta[:histograms][:train]])
        cvae_image_state = inverter(Y_test; maxiter = 1, mode = posterior_mode)

        # Compute decaes on the image data if necessary
        Xs = Dict{Symbol,Dict{Symbol,Any}}()
        Xs[:Y_test]    = Dict(:label => L"Y_{TEST}",       :colour => :grey,   :data => Y_test)
        Xs[:Y_train]   = Dict(:label => L"Y_{TRAIN}",      :colour => :black,  :data => Y_train)
        Xs[:Yhat_mle]  = Dict(:label => L"\hat{Y}_{MLE}",  :colour => :red,    :data => flat_image_to_flat_test(mle_image_state.Y))
        Xs[:Yhat_cvae] = Dict(:label => L"\hat{Y}_{CVAE}", :colour => :blue,   :data => add_noise_instance(derived["ricegen"], cvae_image_state.ν, cvae_image_state.ϵ))
        Xs[:X_decaes]  = Dict(:label => L"X_{DECAES}",     :colour => :orange, :data => flat_test(phys.meta[:decaes][:t2maps][:Y]["decaycurve"]))
        Xs[:X_mle]     = Dict(:label => L"X_{MLE}",        :colour => :green,  :data => flat_image_to_flat_test(mle_image_state.ν))
        Xs[:X_cvae]    = Dict(:label => L"X_{CVAE}",       :colour => :purple, :data => cvae_image_state.ν)

        commonkwargs = Dict{Symbol,Any}(
            # :titlefontsize => 16, :labelfontsize => 14, :xtickfontsize => 12, :ytickfontsize => 12, :legendfontsize => 11,
            :titlefontsize => 10, :labelfontsize => 10, :xtickfontsize => 10, :ytickfontsize => 10, :legendfontsize => 10, #TODO
            :legend => :topright,
        )

        for (key, X) in Xs
            get!(phys.meta[:histograms], :inference, Dict{Symbol, Any}())
            X[:hist] =
                key === :Y_test ? phys.meta[:histograms][dataset] :
                key === :Y_train ? phys.meta[:histograms][:train] :
                (force_histograms || !haskey(phys.meta[:histograms][:inference], key)) ?
                    let
                        @info "Computing signal histogram for $(key) data..."
                        @time signal_histograms(Flux.cpu(X[:data]); edges = Y_train_edges, nbins = nothing)
                    end :
                    phys.meta[:histograms][:inference][key]

            X[:t2dist] =
                key === :Y_test ? flat_test(phys.meta[:decaes][:t2dist][:Y]) :
                key === :Y_train ? flat_train(phys.meta[:decaes][:t2dist][:Y]) :
                key === :X_decaes ? flat_test(phys.meta[:decaes][:t2dist][:Y]) : # decaes signal gives identical t2 distbn by definition, as it consists purely of EPG basis functions
                let
                    if (force_decaes || !haskey(phys.meta[:decaes][:t2maps], key))
                        @info "Computing T2 distribution for $(key) data..."
                        @time t2_distributions!(phys, key => convert(Matrix{Float64}, X[:data]))
                    end
                    flat_test(phys.meta[:decaes][:t2dist][key])
                end

            phys.meta[:histograms][:inference][key] = X[:hist] # update phys metadata
        end

        @info "Plotting histogram distances compared to $dataset data..." # Compare histogram distances for each echo and across all-signal for test data and simulated data
        phist = @time plot(
            map(collect(pairs((; ChiSquared, KLDivergence, CityBlock, Euclidean)))) do (distname, dist)
                echoes = 0:size(phys.images[1].data,4)
                Xplots = [X for (k,X) in Xs if k !== :Y_test]
                logdists = mapreduce(hcat, Xplots) do X
                    (i -> log10(dist(X[:hist][i], Xs[:Y_test][:hist][i]))).(echoes)
                end
                plot(
                    echoes, logdists;
                    label = permutedims(getindex.(Xplots, :label)) .* map(x -> L" ($d_0$ = %$(round(x; sigdigits = 3)))", logdists[1:1,:]),
                    line = (2, permutedims(getindex.(Xplots, :colour))), title = string(distname),
                    commonkwargs...,
                )
            end...;
            commonkwargs...,
        )
        saveplot(phist, "signal-hist-distances")

        @info "Plotting T2 distributions compared to $dataset data..."
        pt2dist = @time let
            Xplots = [X for (k,X) in Xs if k !== :X_decaes]
            # Xplots = [Xs[k] for k ∈ (:Y_test, :Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            T2dists = mapreduce(X -> mean(X[:t2dist]; dims = 2), hcat, Xplots)
            plot(
                1000 .* phys.meta[:decaes][:t2maps][:Y]["t2times"], T2dists;
                label = permutedims(getindex.(Xplots, :label)),
                line = (2, permutedims(getindex.(Xplots, :colour))),
                xscale = :log10,
                xlabel = L"$T_2$ [ms]",
                ylabel = L"$T_2$ Amplitude [a.u.]",
                # title = L"$T_2$-distributions",
                commonkwargs...,
            )
        end
        saveplot(pt2dist, "decaes-T2-distbn")

        @info "Plotting T2 distribution differences compared to $dataset data..."
        pt2diff = @time let
            Xplots = [X for (k,X) in Xs if k ∉ (:X_decaes, :Y_test)]
            # Xplots = [Xs[k] for k ∈ (:Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            T2diffs = mapreduce(X -> mean(X[:t2dist]; dims = 2) .- mean(Xs[:Y_test][:t2dist]; dims = 2), hcat, Xplots)
            logL2 = log10.(sum(abs2, T2diffs; dims = 1))
            plot(
                1000 .* phys.meta[:decaes][:t2maps][:Y]["t2times"], T2diffs;
                label = permutedims(getindex.(Xplots, :label)) .* L" $-$ " .* Xs[:Y_test][:label] .* map(x -> L" ($\log_{10}\ell_2$ = %$(round(x; sigdigits = 3)))", logL2), #TODO
                line = (2, permutedims(getindex.(Xplots, :colour))),
                ylim = (-0.06, 0.1), #TODO
                xscale = :log10,
                xlabel = L"$T_2$ [ms]",
                # ylabel = L"$T_2$ Amplitude [a.u.]", #TODO
                # title = L"$T_2$-distribution Differences",
                commonkwargs...,
            )
        end
        saveplot(pt2diff, "decaes-T2-distbn-diff")

        @info "Plotting signal distributions compared to $dataset data..."
        psignaldist = @time let
            Xplots = [X for (k,X) in Xs if k !== :X_decaes]
            # Xplots = [Xs[k] for k ∈ (:Y_test, :Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            p = plot(;
                xlabel = "Signal magnitude [a.u.]",
                ylabel = "Density [a.u.]",
                commonkwargs...
            )
            for X in Xplots
                plot!(p, normalize(X[:hist][0]); alpha = 0.1, label = X[:label], line = (2, X[:colour]), commonkwargs...)
            end
            p
        end
        saveplot(psignaldist, "decaes-signal-distbn")

        @info "Plotting signal distribution differences compared to $dataset data..."
        psignaldiff = @time let
            Xplots = [X for (k,X) in Xs if k ∉ (:X_decaes, :Y_test)]
            # Xplots = [Xs[k] for k ∈ (:Y_train, :Yhat_mle, :Yhat_cvae)] #TODO
            histdiffs = mapreduce(X -> unitsum(X[:hist][0].weights) .- unitsum(Xs[:Y_test][:hist][0].weights), hcat, Xplots)
            plot(
                Xs[:Y_test][:hist][0].edges[1][2:end], histdiffs;
                label = permutedims(getindex.(Xplots, :label)) .* L" $-$ " .* Xs[:Y_test][:label],
                series = :steppost, line = (2, permutedims(getindex.(Xplots, :colour))),
                xlabel = "Signal magnitude [a.u.]",
                ylim = (-0.002, 0.0035), #TODO
                # ylabel = "Density [a.u.]", #TODO
                commonkwargs...,
            )
        end
        saveplot(psignaldiff, "decaes-signal-distbn-diff")

        saveplot(plot(pt2dist, pt2diff; layout = (1,2), commonkwargs...), "decaes-T2-distbn-and-diff")
        saveplot(plot(psignaldist, psignaldiff, pt2dist, pt2diff; layout = (2,2), commonkwargs...), "decaes-distbn-ensemble")

        @info "Plotting signal distributions compared to $dataset data..." # Compare per-echo and all-signal cdf's of test data and simulated data
        pcdf = plot(; commonkwargs...)
        @time for X in [X for (k,X) in Xs if k ∈ (:Y_test, :Yhat_cvae)]
            plot!(pcdf, discrete_cdf(Flux.cpu(X[:data]))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
            plot!(pcdf, discrete_cdf(reshape(Flux.cpu(X[:data]),1,:))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
        end
        saveplot(pcdf, "signal-cdf-compare")
    end

    function θderived_cpu(θ)
        # named tuple of misc. parameters of interest derived from θ
        map(arr64, θderived(phys, θ))
    end

    function infer_θderived(Y)
        @info "Computing named tuple of θ values, averaging over $naverage samples..."
        θ = @time map(_ -> θderived_cpu(inverter(Y; maxiter = 1, mode = posterior_mode).θ), 1:naverage)
        θ = map((θs...,) -> mean(θs), θ...) # mean over each named tuple field
    end

    let
        mle_sim_data = Glob.readdir(Glob.glob"mle-simulated-mask-data-*.mat", mle_sim_path) |> last |> DECAES.MAT.matread
        mle_sim_results = Glob.readdir(Glob.glob"mle-simulated-mask-results-final-*.mat", mle_sim_path) |> last |> DECAES.MAT.matread

        Ytrue, X̂true, Xtrue, θtrue, Ztrue = getindex.(Ref(mle_sim_data), ("Y", "Xhat", "X", "theta", "Z"))
        θtrue_derived = θtrue |> θderived_cpu
        θmle_derived = mle_sim_results["theta"] |> θderived_cpu

        @info "Computing DECAES inference error..."
        if (force_decaes || !haskey(phys.meta[:decaes][:t2maps], :Yhat_cvae_decaes))
            @time t2_distributions!(phys, :Yhat_cvae_decaes => convert(Matrix{Float64}, X̂true))
        end
        @info 0, :decaes, [
            L"\alpha" => mean(abs, θtrue_derived.alpha - vec(phys.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["alpha"])),
            L"\bar{T}_2" => mean(abs, θtrue_derived.T2bar - vec(phys.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["ggm"])),
            L"T_{2,SGM}" => mean(abs, filter(!isnan, θtrue_derived.T2sgm - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sgm"]))), # "sgm" is set to NaN if all T2 components within SPWin are zero; be generous with error measurement
            L"T_{2,MGM}" => mean(abs, filter(!isnan, θtrue_derived.T2mgm - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["mgm"]))), # "mgm" is set to NaN if all T2 components within MPWin are zero; be generous with error measurement
            L"MWF" => mean(abs, filter(!isnan, θtrue_derived.mwf - vec(phys.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sfr"]))),
        ]

        @info "Computing MLE inference error..."
        @info 0, :mle, [lab =>round(mean(abs, θt - θi); sigdigits = 3) for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θmle_derived)]

        for mode in [:mean], maxiter in [1,2,5]
            inf_time = @elapsed cvae_state = inverter(X̂true |> to32; maxiter, mode, verbose = false)
            θcvae_derived = cvae_state.θ |> θderived_cpu
            @info maxiter, mode, inf_time, [lab => round(mean(abs, θt - θi); sigdigits = 3) for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θcvae_derived)]
        end
    end

    # Heatmaps
    Y = phys.images[1].data[:,:,zslices,:] # (nx, ny, nslice, nTE)
    Islices = findall(!isnan, Y[..,1]) # entries within Y mask
    Imaskslices = filter(I -> I[3] ∈ zslices, phys.images[1].indices[:mask])
    makemaps(x) = (out = fill(NaN, size(Y)[1:3]); out[Islices] .= Flux.cpu(x); return permutedims(out, (2,1,3)))

    θcvae = infer_θderived(permutedims(Y[Islices,:]) |> to32)
    θmle = θderived_cpu(flat_image_to_flat_indices(mle_image_state.θ, Imaskslices))

    # DECAES heatmaps
    @time let
        θdecaes = (
            alpha   = (phys.meta[:decaes][:t2maps][:Y]["alpha"], L"\alpha",     (50.0, 180.0)),
            T2bar   = (phys.meta[:decaes][:t2maps][:Y]["ggm"],   L"\bar{T}_2",  (0.0, 0.25)),
            T2sgm   = (phys.meta[:decaes][:t2parts][:Y]["sgm"],  L"T_{2,SGM}",  (0.0, 0.1)),
            T2mgm   = (phys.meta[:decaes][:t2parts][:Y]["mgm"],  L"T_{2,MGM}",  (0.0, 1.0)),
            mwf     = (phys.meta[:decaes][:t2parts][:Y]["sfr"],  L"MWF",        (0.0, 0.4)),
        )
        for (θname, (θk, θlabel, θbd)) in pairs(θdecaes), (j,zj) in enumerate(zslices)
            pyheatmap(permutedims(θk[:,:,zj]); title = θlabel * " (slice $zj)", clim = θbd, filename = joinpath(mkpath(joinpath(savefolder, "decaes")), "$θname-$zj"), savetypes)
        end
    end

    # CVAE and MLE heatmaps
    for (θfolder, θ) ∈ [:cvae => θcvae, :mle => θmle]
        @info "Plotting heatmap plots for mean θ values..."
        @time let
            for (k, ((θname, θk), θlabel, θbd)) in enumerate(zip(pairs(θ), θderivedlabels(phys), θderivedbounds(phys)))
                θmaps = makemaps(θk)
                for (j,zj) in enumerate(zslices)
                    pyheatmap(θmaps[:,:,j]; title = θlabel * " (slice $zj)", clim = θbd, filename = joinpath(mkpath(joinpath(savefolder, string(θfolder))), "$θname-$zj"), savetypes)
                end
            end
        end

        @info "Plotting T2-distribution over test data..."
        @time let
            T2 = 1000 .* vcat(θ.T2short, θ.T2long)
            A = vcat(θ.Ashort, θ.Along)
            p = plot(
                # bin_edges(T2, A, exp.(range(log.(phys.T2bd)...; length = 100)))...;
                bin_sorted(T2, A; binsize = 100)...;
                label = "T2 Distribution", ylabel = "T2 Amplitude [a.u.]", xlabel = "T2 [ms]",
                xscale = :log10, xlim = 1000 .* phys.T2bd, xticks = 10 .^ (0.5:0.25:3),
            )
            saveplot(p, "T2distbn-$(zslices[1])-$(zslices[end])", joinpath(savefolder, string(θfolder)))
        end
    end
end

####
#### MCMC inference
####

#=
TODO: implement MCMC

#=
Turing.@model turing_signal_model(
        y,
        correction_and_noiselevel,
    ) = begin
    freq   ~ Uniform(1/64,  1/32)
    phase  ~ Uniform( 0.0,  pi/2)
    offset ~ Uniform( 0.25,  0.5)
    amp    ~ Uniform( 0.1,  0.25)
    tconst ~ Uniform(16.0, 128.0)
    # logeps ~ Uniform(-4.0,  -2.0)
    # epsilon = 10^logeps

    # Compute toy signal model without noise
    x = toy_signal_model([freq, phase, offset, amp, tconst], nothing, 4)
    yhat, ϵhat = correction_and_noiselevel(x)

    # Model noise as Rician
    for i in 1:length(y)
        # ν, σ = x[i], epsilon
        ν, σ = yhat[i], ϵhat[i]
        y[i] ~ Rician(ν, σ)
    end
end
=#

function toy_theta_mcmc_inference(
        y::AbstractVector,
        correction_and_noiselevel,
        callback = (y, chain) -> true,
    )
    model = function (x)
        xhat, ϵhat = correction_and_noiselevel(x)
        yhat = rand.(Rician.(xhat, ϵhat))
        return yhat
    end
    res = signal_loglikelihood_inference(y, nothing, model)
    theta0 = best_candidate(res)
    while true
        chain = sample(turing_signal_model(y, correction_and_noiselevel), NUTS(), 1000; verbose = true, init_theta = theta0)
        # chain = psample(turing_signal_model(y, correction_and_noiselevel), NUTS(), 1000, 3; verbose = true, init_theta = theta0)
        callback(y, chain) && return chain
    end
end

function toy_theta_mcmc_inference(Y::AbstractMatrix, args...; kwargs...)
    tasks = map(1:size(Y,2)) do j
        Threads.@spawn signal_loglikelihood_inference(Y[:,j], initial_guess, args...; kwargs...)
    end
    return map(Threads.fetch, tasks)
end

TODO: implement MCMC
=#
