####
#### MLE inference
####

function mle_biexp_epg_noise_only(
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
        initial_ϵ     = nothing,
        initial_s     = nothing;
        batch_size    = 128 * Threads.nthreads(),
        verbose       = true,
        checkpoint    = false,
        dryrun        = false,
        dryrunsamples = nothing,
        dryrunshuffle = true,
        savefolder    = nothing,
        opt_alg       = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args      = Dict{Symbol,Any}(),
    )
    @assert dryrun || (savefolder !== nothing)
    matsavetime = getnow()
    matsave(filename, data) = DECAES.MAT.matwrite(joinpath(mkpath(savefolder), filename * "-" * matsavetime * ".mat"), data)

    dryrun && (dryrunsamples !== nothing) && let
        I = dryrunshuffle ?
            sample(MersenneTwister(0), 1:size(Y,2), dryrunsamples; replace = dryrunsamples > size(Y,2)) :
            1:min(size(Y,2), dryrunsamples)
        X = X[:,I]
        Y = Y[:,I]
        (initial_ϵ !== nothing) && (initial_ϵ = initial_ϵ[:,I])
        (initial_s !== nothing) && (initial_s = initial_s[:,I])
    end

    (initial_ϵ === nothing) && (initial_ϵ = sqrt.(mean(abs2, X .- Y; dims = 1)))
    (initial_s === nothing) && (initial_s = inv.(maximum(_rician_mean_cuda.(X, initial_ϵ); dims = 1))) # ones_similar(X, 1, size(X,2))
    @assert size(X) == size(Y) && size(initial_ϵ) == size(initial_s) == (1, size(X,2))

    initial_loss  = -sum(_rician_logpdf_cuda.(Y, initial_s .* X, initial_s .* initial_ϵ); dims = 1)
    initial_guess = (
        ϵ = initial_ϵ |> arr64,
        s = initial_s |> arr64,
        ℓ = initial_loss |> arr64,
    )
    X, Y = (X, Y) .|> arr64

    work_spaces = map(1:Threads.nthreads()) do _
        (
            Xj  = zeros(size(X,1)),
            Yj  = zeros(size(Y,1)),
            x0  = zeros(2),
            opt = let
                opt = NLopt.Opt(opt_alg, 2)
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
        logepsilon = fill(NaN, size(Y,2)),
        logscale   = fill(NaN, size(Y,2)),
        loss       = fill(NaN, size(Y,2)),
        retcode    = fill(NaN, size(Y,2)),
        numevals   = fill(NaN, size(Y,2)),
        solvetime  = fill(NaN, size(Y,2)),
    )

    function f(work, x::Vector{Float64})
        δ₀ = sqrt(eps(Float64))
        @inbounds begin
            ϵ  = exp(x[1])
            s  = exp(x[2])
            ϵi = max(s * ϵ, δ₀)
            ℓ  = 0.0
            for i in eachindex(work.Xj)
                νi = max(s * work.Xj[i], δ₀)
                ℓ -= _rician_logpdf_cuda(work.Yj[i], νi, ϵi) # Rician negative log likelihood
            end
            return ℓ
        end
    end

    function fg!(work, x::Vector{Float64}, g::Vector{Float64})
        if length(g) > 0
            simple_fd_gradient!(g, y -> f(work, y), x)
        else
            f(work, x)
        end
    end

    #= Benchmarking
    work_spaces[1].Xj .= X[:,1]
    work_spaces[1].Yj .= Y[:,1]
    work_spaces[1].x0 .= log.(vcat(initial_guess.ϵ[1:1,1], initial_guess.s[1:1,1]))
    @info "Calling function..."; l = f( work_spaces[1], work_spaces[1].x0 ); @show l
    @info "Calling gradient..."; g = zeros(2); l = fg!( work_spaces[1], work_spaces[1].x0, g ); @show l, g
    @info "Timing function..."; @btime( $f( $( work_spaces[1] ), $( work_spaces[1].x0 ) ) )
    @info "Timing gradient..."; @btime( $fg!( $( work_spaces[1] ), $( work_spaces[1].x0 ), $( g ) ) )
    =#

    start_time = time()
    batches = Iterators.partition(1:size(Y,2), batch_size)
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Xj               .= X[:,j]
                work.Yj               .= Y[:,j]
                work.x0[1]             = log(initial_guess.ϵ[1,j])
                work.x0[2]             = log(initial_guess.s[1,j])
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.logepsilon[j]  = minx[1]
                results.logscale[j]    = minx[2]
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
        savetime       = @elapsed !dryrun && checkpoint && matsave("mle-$data_source-$data_subset-results-checkpoint", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # checkpoint progress
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- save: $(round(savetime; digits = 2))s" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))"
    end

    !dryrun && matsave("mle-$data_source-$data_subset-results-final", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # save final results
    BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end

function _test_mle_biexp_epg_noise_only(phys::BiexpEPGModel, derived; nsamples = 10240, init_epsilon = false)
    θ = sample(derived["genatr_theta_prior"], nsamples)
    Z = sample(derived["genatr_latent_prior"], nsamples)
    X = signal_model(phys, θ)
    X̂ = sampleX̂(models["genatr"], X, Z) #TODO

    initial_ϵ = init_epsilon ?
        rician_params(models["genatr"], X, Z)[2][1:1,:] : # use true answer as initial guess (for sanity check)
        sqrt.(mean(abs2, X .- X̂; dims = 1))
    initial_s = inv.(maximum(_rician_mean_cuda.(X, initial_ϵ); dims = 1)) # ones_similar(X, 1, size(X,2))

    init, res = mle_biexp_epg_noise_only(
        X, X̂, initial_ϵ, initial_s;
        verbose       = true,
        checkpoint    = false,
        dryrun        = true,
        dryrunsamples = nsamples,
        dryrunshuffle = false,
    )

    ϵ_true = Z .+ log.(models["genatr"].noisescale(X)); # logϵ = log(exp(Z) * noisescale(X)) = Z + log(noisescale(X))
    println("initial log epsilon:"); display(log.(init.ϵ))
    println("final log epsilon:"); display(res.logepsilon')
    println("true log epsilon:"); display(ϵ_true); display(mean(abs, arr_similar(ϵ_true, res.logepsilon') .- ϵ_true))
    println("")
    println("initial log scale:"); display(log.(init.s))
    println("final log scale:"); display(res.logscale')
    println("")
    println("initial loss:"); display(init.ℓ)
    println("final loss:"); display(res.loss')
    println("")
    println("return codes:"); display(res.retcode')

    return @ntuple(θ, Z, X, X̂, init, res)
end

function _test_noiselevel()
    G = models["genatr"]
    θ = sample(derived["genatr_theta_prior"], 10240)
    Z = sample(derived["genatr_latent_prior"], 10240)
    X = signal_model(phys, θ)
    let
        ν, ϵ = rician_params(G, X, Z)
        X̂ = add_noise_instance(G, X, ϵ)
        # vcat(exp.(Z) .* G.noisescale(X), ϵ) |> display
        NegLogLikelihood(G, X̂, ν, ϵ) |> display
        NegLogLikelihood(G, X̂, ν, ϵ) |> mean |> display
    end
    let
        δ, ϵ = correction_and_noiselevel(G, X, Z)
        ν = add_correction(G, X, δ)
        X̂ = add_noise_instance(G, X, ϵ)
        # vcat(exp.(Z) .* G.noisescale(X), ϵ) |> display
        reshape(NegLogLikelihood(G, X̂, ν, ϵ), 1, :) |> display
        reshape(NegLogLikelihood(G, X̂, ν, ϵ), 1, :) |> mean |> display
    end
    let
        ν, ϵ = rician_params(G, X, Z)
        X̂ = add_noise_instance(G, X, ϵ)
        X̂meta = MetaCPMGSignal(phys, img, X̂)
        fit_state = fit_cvae(X̂meta; marginalize_Z = false)
        display(vcat(Z, fit_state.Z))
        Z1, Z2, Z3 = fit_state.Z, fit_state.Z .- log.(G.noisescale(X)), fit_state.Z .+ log.(G.noisescale(X))
        ϵ1, ϵ2, ϵ3 = exp.(Z1), exp.(Z2), exp.(Z3)
        vcat(
            quantile(abs.(Z1 .- Z) |> vec |> arr64, 0.1:0.1:0.9)',
            quantile(abs.(Z2 .- Z) |> vec |> arr64, 0.1:0.1:0.9)',
            quantile(abs.(Z3 .- Z) |> vec |> arr64, 0.1:0.1:0.9)',
        ) |> display
        vcat(
            reshape(NegLogLikelihood(G, X̂, ν, ϵ1), 1, :),
            reshape(NegLogLikelihood(G, X̂, ν, ϵ2), 1, :),
            reshape(NegLogLikelihood(G, X̂, ν, ϵ3), 1, :),
        ) |> display
        reshape(NegLogLikelihood(G, X̂, ν, ϵ1), 1, :) |> mean |> display
        reshape(NegLogLikelihood(G, X̂, ν, ϵ2), 1, :) |> mean |> display
        reshape(NegLogLikelihood(G, X̂, ν, ϵ3), 1, :) |> mean |> display
        display(vcat(fit_state.ℓ))
        display(mean(fit_state.ℓ))
    end
end

function mle_biexp_epg(
        phys::BiexpEPGModel,
        models,
        derived,
        img::CPMGImage;
        data_source   = :image, # One of :image, :simulated
        data_subset   = :mask,  # One of :mask, :val, :train, :test
        batch_size    = 128 * Threads.nthreads(),
        initial_iter  = 10,
        verbose       = true,
        checkpoint    = false,
        dryrun        = false,
        dryrunsamples = batch_size,
        dryrunshuffle = true,
        savefolder    = nothing,
        opt_alg       = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args      = Dict{Symbol,Any}(),
    )
    @assert data_source ∈ (:image, :simulated)
    @assert data_subset ∈ (:mask, :val, :train, :test)
    @assert dryrun || (savefolder !== nothing)
    matsavetime = getnow()
    matsave(filename, data) = DECAES.MAT.matwrite(joinpath(mkpath(savefolder), filename * "-" * matsavetime * ".mat"), data)

    batched_posterior_state(Ymeta) = mapreduce(
            (x,y) -> map(hcat, x, y),
            enumerate(Iterators.partition(1:size(signal(Ymeta),2), batch_size)),
        ) do (batchnum, batch)
        posterior_state(
            phys,
            models["genatr"],
            derived["cvae"],
            Ymeta[:,batch];
            miniter = 1,
            maxiter = initial_iter,
            alpha   = 0.0,
            verbose = false,
            mode    = :maxlikelihood,
        )
    end

    # MLE for whole image of simulated data
    Y, Ymeta = let
        image_indices = img.indices[data_subset]
        if dryrun && (dryrunsamples !== nothing)
            I = dryrunshuffle ?
                sample(MersenneTwister(0), 1:length(image_indices), dryrunsamples; replace = dryrunsamples > length(image_indices)) :
                1:min(length(image_indices), dryrunsamples)
            image_indices = image_indices[I]
        end
        image_data = img.data[image_indices, :] |> to32 |> permutedims

        if data_source === :image
            !dryrun && matsave("mle-$data_source-$data_subset-data", Dict{String,Any}("Y" => arr64(image_data)))
            arr64(image_data), MetaCPMGSignal(phys, img, image_data)
        else # data_source === :simulated
            mock_image_state = batched_posterior_state(MetaCPMGSignal(phys, img, image_data))
            mock_image_data  = sampleX̂(models["genatr"], mock_image_state.X, mock_image_state.Z)
            !dryrun && matsave("mle-$data_source-$data_subset-data", Dict{String,Any}("X" => arr64(mock_image_state.X), "theta" => arr64(mock_image_state.θ), "Z" => arr64(mock_image_state.Z), "Xhat" => arr64(mock_image_data), "Y" => arr64(image_data)))
            arr64(mock_image_data), MetaCPMGSignal(phys, img, mock_image_data)
        end
    end

    initial_guess = batched_posterior_state(Ymeta)
    initial_guess = setindex!!(initial_guess, exp.(mean(log.(initial_guess.ϵ); dims = 1)), :ϵ)
    initial_guess = setindex!!(initial_guess, inv.(maximum(_rician_mean_cuda.(initial_guess.X, initial_guess.ϵ); dims = 1)), :s)
    initial_guess = map(arr64, initial_guess)

    lower_bounds  = [θmarginalized(phys, θlower(phys)); -Inf; -Inf] |> arr64
    upper_bounds  = [θmarginalized(phys, θupper(phys)); +Inf; +Inf] |> arr64

    function θ_to_x(work, θ::Union{<:AbstractVector{Float64},<:NTuple{<:Any,Float64}})
        @inbounds begin
            @unpack logτ0, logτ1, logτ0′, logτ1′ = work
            α, β, η, δ1, δ2 = θ[1], θ[2], θ[3], θ[4], θ[5]
            t   = 1e-12
            δ1′ = clamp(logτ0 + (logτ1 - logτ0) * δ1, logτ0′ + t, logτ1′ - t)
            δ2′ = clamp(logτ0 + (logτ1 - logτ0) * (δ1 + δ2 * (1 - δ1)), logτ0′ + t, logτ1′ - t)
            z1  = (δ1′ - logτ0′) / (logτ1′ - logτ0′)
            z2  = ((δ2′ - logτ0′) / (logτ1′ - logτ0′) - z1) / (1 - z1)
            return α, β, η, z1, z2
        end
    end

    function x_to_θ(work, x::Union{<:AbstractVector{Float64},<:NTuple{<:Any,Float64}})
        @inbounds begin
            @unpack logτ0, logτ1, logτ0′, logτ1′, δ0 = work
            α, β, η, z1, z2 = x[1], x[2], x[3], x[4], x[5]
            δ1 = ((logτ0′ - logτ0) + (logτ1′ - logτ0′) * z1) / (logτ1 - logτ0)
            δ2 = (((logτ0′ - logτ0) + (logτ1′ - logτ0′) * (z1 + z2 * (1 - z1))) / (logτ1 - logτ0) - δ1) / (1 - δ1)
            return α, β, η, δ1, δ2, δ0
        end
    end

    work_spaces = map(1:Threads.nthreads()) do _
        @unpack TEbd, T2bd = phys
        (
            Y   = zeros(nsignal(img)) |> arr64,
            x0  = zeros(nmarginalized(phys) + 2) |> arr64,
            logτ0  = log(T2bd[1] / TEbd[2]) |> Float64,
            logτ1  = log(T2bd[2] / TEbd[1]) |> Float64,
            logτ0′ = log(T2bd[1] / echotime(img)) |> Float64,
            logτ1′ = log(T2bd[2] / echotime(img)) |> Float64,
            δ0  = initial_guess.θ[end,1] |> Float64,
            epg = BiexpEPGModelWork(phys),
            opt = let
                opt = NLopt.Opt(opt_alg, nmarginalized(phys) + 2)
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
        theta      = fill(NaN, ntheta(phys), size(Y,2)),
        logepsilon = fill(NaN, size(Y,2)),
        logscale   = fill(NaN, size(Y,2)),
        loss       = fill(NaN, size(Y,2)),
        retcode    = fill(NaN, size(Y,2)),
        numevals   = fill(NaN, size(Y,2)),
        solvetime  = fill(NaN, size(Y,2)),
    )

    function f(work, x::Vector{Float64})
        @inbounds begin
            l  = sqrt(eps(Float64))
            θ  = x_to_θ(work, x) # θ = α, β, η, δ1, δ2, δ0
            ϵ  = exp(x[end-1])
            s  = exp(x[end])
            ψ  = θmodel(phys, θ...) # ψ = alpha, refcon, T2short, T2long, Ashort, Along, T1, TE
            X  = _signal_model_f64(phys, work.epg, ψ)
            sX = 0.0
            @simd for i in eachindex(X)
                sX = max(X[i], sX)
            end
            sX = s / sX # normalize X to maximum 1 and scale by s
            ϵi = max(s * ϵ, l) # hoist outside loop
            ℓ  = 0.0
            @simd for i in eachindex(work.Y)
                νi = max(sX * X[i], l)
                ℓ -= _rician_logpdf_cuda(work.Y[i], νi, ϵi) # Rician negative log likelihood
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
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Y                .= Y[:,j]
                work.x0[1:end-2]      .= θ_to_x(work, view(initial_guess.θ, :, j))
                work.x0[end-1]         = log(initial_guess.ϵ[j])
                work.x0[end]           = log(initial_guess.s[j])
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                initial_guess.ℓ[j] = f(work, work.x0) #TODO: can save a small amount of time by deleting this, but probably worth it, since previous ℓ[j] is an approximation from posterior_state

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.theta[:,j]    .= x_to_θ(work, minx)
                results.logepsilon[j]  = minx[end-1]
                results.logscale[j]    = minx[end]
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
        savetime       = @elapsed !dryrun && checkpoint && matsave("mle-$data_source-$data_subset-results-checkpoint", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # checkpoint progress
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- save: $(round(savetime; digits = 2))s" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))"
    end

    !dryrun && matsave("mle-$data_source-$data_subset-results-final", Dict{String,Any}(string(k) => copy(v) for (k,v) in pairs(results))) # save final results
    BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end

function _test_mle_biexp_epg(phys::BiexpEPGModel, derived, img::CPMGImage; dryrunsamples = 10240, kwargs...)
    init, res = mle_biexp_epg(
        phys,
        nothing,
        derived,
        img;
        verbose       = true,
        checkpoint    = false,
        dryrun        = true,
        dryrunsamples = dryrunsamples,
        dryrunshuffle = false,
        kwargs...,
    )

    println("initial theta:"); display(init.θ)
    println("final theta:"); display(res.theta)
    println("mean abs diff:"); display(mean(abs, init.θ .- res.theta; dims = 2))
    println("initial log epsilon:"); display(log.(init.ϵ[1:1,:]))
    println("final log epsilon:"); display(res.logepsilon')
    println("mean abs diff:"); display(mean(abs, log.(init.ϵ[1:1,:]) .- res.logepsilon'))
    println("")
    println("initial log scale:"); display(log.(init.s[1:1,:]))
    println("final log scale:"); display(res.logscale')
    println("mean abs diff:"); display(mean(abs, log.(init.s[1:1,:]) .- res.logscale'))
    println("")
    println("initial loss:"); display(init.ℓ)
    println("final loss:"); display(res.loss')
    println("mean abs diff:"); display(mean(abs, init.ℓ .- res.loss'))
    println("")
    println("return codes:"); display(res.retcode')

    return init, res
end

####
#### Model evaluation
####

function eval_mri_model(
        phys::BiexpEPGModel,
        models,
        derived,
        img::CPMGImage;
        slices = 24:24,
        slicedim = 3,
        naverage = 10,
        savefolder = nothing,
        savetypes = [".png"],
        mle_image_path = nothing,
        mle_sim_path = nothing,
        batch_size = nothing,
        force_decaes = false,
        force_histograms = false,
        posterior_mode = :maxlikelihood,
        quiet = false,
        dataset = :val, # :val or (for final model comparison) :test
    )

    inverter(Y; kwargs...) = mapreduce((x,y) -> map(hcat, x, y), enumerate(Iterators.partition(1:size(Y,2), batch_size))) do (batchnum, batch)
        posterior_state(
            phys, models["genatr"], derived["cvae"], MetaCPMGSignal(phys, img, Y[:,batch]);
            verbose = false, alpha = 0.0, miniter = 1, maxiter = naverage, mode = posterior_mode, kwargs...
        )
    end
    saveplot(p, name, folder = savefolder) = map(suf -> savefig(p, joinpath(mkpath(folder), name * suf)), savetypes)

    flat_test(x) = flat_indices(x, img.indices[dataset])
    flat_train(x) = flat_indices(x, img.indices[:train])
    flat_indices(x, indices) =
        x isa AbstractMatrix ? (@assert(size(x,2) == length(indices)); return x) : # matrix with length(indices) columns
        x isa AbstractTensor4D ?
            (size(x)[1:3] == (length(indices), 1, 1)) ? permutedims(reshape(x, :, size(x,4))) : # flattened 4D array with first three dimensions (length(indices), 1, 1)
            (size(x)[1:3] == size(img.data)[1:3]) ? permutedims(x[indices,:]) : # 4D array with first three dimensions equal to image size
            error("4D array has wrong shape") :
        error("x must be an $AbstractMatrix or an $AbstractTensor4D")

    flat_image_to_flat_test(x) = flat_image_to_flat_indices(x, img.indices[dataset])
    flat_image_to_flat_train(x) = flat_image_to_flat_indices(x, img.indices[:train])
    function flat_image_to_flat_indices(x, indices)
        _x = similar(x, size(x,1), size(img.data)[1:3]...)
        _x[:, img.indices[:mask]] = x
        return _x[:, indices]
    end

    # Compute decaes on the image data if necessary
    if !haskey(img.meta, :decaes)
        @info "Recomputing T2 distribution for image data..."
        @time t2_distributions!(img)
    end

    mle_image_state = let
        mle_image_results = Glob.readdir(Glob.glob"mle-image-mask-results-final-*.mat", mle_image_path) |> only |> DECAES.MAT.matread
        θ = mle_image_results["theta"] |> to32
        ϵ = reshape(exp.(mle_image_results["logepsilon"] |> to32), 1, :)
        ℓ = reshape(mle_image_results["loss"] |> to32, 1, :) # negative log-likelihood loss
        X = signal_model(phys, θ)[1:nsignal(img), :]
        ν, δ, Z = X, nothing, nothing
        Y = add_noise_instance(phys, X, ϵ)
        (; Y, θ, Z, X, δ, ϵ, ν, ℓ)
    end

    let
        Y_test = img.partitions[dataset] |> to32
        Y_train = img.partitions[:train] |> to32
        Y_train_edges = Dict([k => v.edges[1] for (k,v) in img.meta[:histograms][:train]])
        cvae_image_state = inverter(Y_test; maxiter = 1, mode = posterior_mode)

        # Compute decaes on the image data if necessary
        Xs = Dict{Symbol,Dict{Symbol,Any}}()
        Xs[:Y_test]    = Dict(:label => L"Y_{TEST}",       :colour => :grey,   :data => Y_test)
        Xs[:Y_train]   = Dict(:label => L"Y_{TRAIN}",      :colour => :black,  :data => Y_train)
        Xs[:Yhat_mle]  = Dict(:label => L"\hat{Y}_{MLE}",  :colour => :red,    :data => flat_image_to_flat_test(mle_image_state.Y))
        Xs[:Yhat_cvae] = Dict(:label => L"\hat{Y}_{CVAE}", :colour => :blue,   :data => add_noise_instance(models["genatr"], cvae_image_state.ν, cvae_image_state.ϵ))
        Xs[:X_decaes]  = Dict(:label => L"X_{DECAES}",     :colour => :orange, :data => flat_test(img.meta[:decaes][:t2maps][:Y]["decaycurve"]))
        Xs[:X_mle]     = Dict(:label => L"X_{MLE}",        :colour => :green,  :data => flat_image_to_flat_test(mle_image_state.ν))
        Xs[:X_cvae]    = Dict(:label => L"X_{CVAE}",       :colour => :purple, :data => cvae_image_state.ν)

        commonkwargs = Dict{Symbol,Any}(
            # :titlefontsize => 16, :labelfontsize => 14, :xtickfontsize => 12, :ytickfontsize => 12, :legendfontsize => 11,
            :titlefontsize => 10, :labelfontsize => 10, :xtickfontsize => 10, :ytickfontsize => 10, :legendfontsize => 10, #TODO
            :legend => :topright,
        )

        for (key, X) in Xs
            get!(img.meta[:histograms], :inference, Dict{Symbol, Any}())
            X[:hist] =
                key === :Y_test ? img.meta[:histograms][dataset] :
                key === :Y_train ? img.meta[:histograms][:train] :
                (force_histograms || !haskey(img.meta[:histograms][:inference], key)) ?
                    let
                        @info "Computing signal histogram for $(key) data..."
                        @time signal_histograms(cpu(X[:data]); edges = Y_train_edges, nbins = nothing)
                    end :
                    img.meta[:histograms][:inference][key]

            X[:t2dist] =
                key === :Y_test ? flat_test(img.meta[:decaes][:t2dist][:Y]) :
                key === :Y_train ? flat_train(img.meta[:decaes][:t2dist][:Y]) :
                key === :X_decaes ? flat_test(img.meta[:decaes][:t2dist][:Y]) : # decaes signal gives identical t2 distbn by definition, as it consists purely of EPG basis functions
                let
                    if (force_decaes || !haskey(img.meta[:decaes][:t2maps], key))
                        @info "Computing T2 distribution for $(key) data..."
                        @time t2_distributions!(img, key => convert(Matrix{Float64}, X[:data]))
                    end
                    flat_test(img.meta[:decaes][:t2dist][key])
                end

            img.meta[:histograms][:inference][key] = X[:hist] # update img metadata
        end

        @info "Plotting histogram distances compared to $dataset data..." # Compare histogram distances for each echo and across all-signal for test data and simulated data
        phist = @time plot(
            map(collect(pairs((; ChiSquared, CityBlock, Euclidean)))) do (distname, dist) # KLDivergence
                echoes = 0:size(img.data,4)
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
                1000 .* img.meta[:decaes][:t2maps][:Y]["t2times"], T2dists;
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
                1000 .* img.meta[:decaes][:t2maps][:Y]["t2times"], T2diffs;
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
            plot!(pcdf, discrete_cdf(cpu(X[:data]))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
            plot!(pcdf, discrete_cdf(reshape(cpu(X[:data]),1,:))...; line = (1, X[:colour]), legend = :none, commonkwargs...)
        end
        saveplot(pcdf, "signal-cdf-compare")
    end

    function θderived_cpu(θ)
        # named tuple of misc. parameters of interest derived from θ
        map(arr64, θderived(phys, img, θ |> to32))
    end

    function infer_θderived(Y)
        @info "Computing named tuple of θ values, averaging over $naverage samples..."
        θ = @time map(_ -> θderived_cpu(inverter(Y; maxiter = 1, mode = posterior_mode).θ), 1:naverage)
        θ = map((θs...,) -> mean(θs), θ...) # mean over each named tuple field
    end

    # Heatmaps
    let
        get_slice(x, sj) = slicedim == 1 ? x[sj,..] : slicedim == 2 ? x[:,sj,..] : x[:,:,sj,..]
        orient_slice(x) = slicedim == 1 ? x : slicedim == 2 ? x[end:-1:1,:] : permutedims(x)
        Y = get_slice(img.data, slices) # (nx, ny, nslice, nTE)
        Islices = findall(!isnan, Y[..,1]) # entries within Y mask
        Imaskslices = filter(I -> I[slicedim] ∈ slices, img.indices[:mask])
        fill_maps(x) = (out = fill(NaN, size(Y)[1:3]); out[Islices] .= cpu(x); return out)

        θcvae = infer_θderived(permutedims(Y[Islices,:]) |> to32)
        θmle = θderived_cpu(flat_image_to_flat_indices(mle_image_state.θ, Imaskslices))

        # DECAES heatmaps
        @time let
            θdecaes = (
                alpha   = (img.meta[:decaes][:t2maps][:Y]["alpha"],       L"\alpha",    (50.0, 180.0)),
                T2bar   = (img.meta[:decaes][:t2maps][:Y]["ggm"],         L"\bar{T}_2", (0.0, 0.25)),
                T2sgm   = (img.meta[:decaes][:t2parts][:Y]["sgm"],        L"T_{2,SGM}", (0.0, 0.1)),
                T2mgm   = (img.meta[:decaes][:t2parts][:Y]["mgm"],        L"T_{2,MGM}", (0.0, 1.0)),
                mwf     = (100 .* img.meta[:decaes][:t2parts][:Y]["sfr"], L"MWF",       (0.0, 40.0)),
            )
            for (θname, (θk, θlabel, θbd)) in pairs(θdecaes), (j,sj) in enumerate(slices)
                pyheatmap(orient_slice(get_slice(θk, sj)); title = θlabel * " (slice $sj)", clim = θbd, axis = :off, aspect = 4/3, filename = joinpath(mkpath(joinpath(savefolder, "decaes")), "$θname-$sj"), savetypes)
            end
        end

        # CVAE and MLE heatmaps
        for (θfolder, θ) ∈ [:cvae => θcvae, :mle => θmle]
            @info "Plotting heatmap plots for mean θ values..."
            @time let
                for (k, ((θname, θk), θlabel, θbd)) in enumerate(zip(pairs(θ), θderivedlabels(phys), θderivedbounds(phys)))
                    θmaps = fill_maps(θk)
                    for (j,sj) in enumerate(slices)
                        pyheatmap(orient_slice(get_slice(θmaps, j)); title = θlabel * " (slice $sj)", clim = θbd, axis = :off, aspect = 4/3, filename = joinpath(mkpath(joinpath(savefolder, string(θfolder))), "$θname-$sj"), savetypes)
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
                saveplot(p, "T2distbn-$(slices[1])-$(slices[end])", joinpath(savefolder, string(θfolder)))
            end
        end
    end

    # Error tables
    let
        mle_sim_data = Glob.readdir(Glob.glob"mle-simulated-mask-data-*.mat", mle_sim_path) |> only |> DECAES.MAT.matread
        mle_sim_results = Glob.readdir(Glob.glob"mle-simulated-mask-results-final-*.mat", mle_sim_path) |> only |> DECAES.MAT.matread

        Ytrue, X̂true, Xtrue, θtrue, Ztrue = getindex.(Ref(mle_sim_data), ("Y", "Xhat", "X", "theta", "Z"))
        θtrue_derived = θtrue |> θderived_cpu
        θmle_derived = mle_sim_results["theta"] |> θderived_cpu

        all_errors = Any[]
        all_row_labels = [θderivedlabels(phys); "Time"]
        all_row_units  = [θderivedunits(phys); "min"]

        @info "Computing DECAES inference error..."
        decaes_errors = Dict{AbstractString, Float64}(all_row_labels .=> NaN)
        if (force_decaes || !haskey(img.meta[:decaes][:t2maps], :Yhat_cvae_decaes))
            decaes_errors["Time"]  = @elapsed t2_distributions!(img, :Yhat_cvae_decaes => convert(Matrix{Float64}, X̂true))
            decaes_errors["Time"] /= 60 # convert sec => min
        end
        decaes_errors[L"\alpha"] = mean(abs, θtrue_derived.alpha - vec(img.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["alpha"]))
        decaes_errors[L"\bar{T}_2"] = mean(abs, θtrue_derived.T2bar - vec(img.meta[:decaes][:t2maps][:Yhat_cvae_decaes]["ggm"]))
        decaes_errors[L"T_{2,SGM}"] = mean(abs, filter(!isnan, θtrue_derived.T2sgm - vec(img.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sgm"]))) # "sgm" is set to NaN if all T2 components within SPWin are zero; be generous with error measurement
        decaes_errors[L"T_{2,MGM}"] = mean(abs, filter(!isnan, θtrue_derived.T2mgm - vec(img.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["mgm"]))) # "mgm" is set to NaN if all T2 components within MPWin are zero; be generous with error measurement
        decaes_errors[L"MWF"] = mean(abs, filter(!isnan, θtrue_derived.mwf - 100 .* vec(img.meta[:decaes][:t2parts][:Yhat_cvae_decaes]["sfr"])))
        push!(all_errors, "DECAES" => decaes_errors)

        @info "Computing MLE inference error..."
        mle_errors = Dict{AbstractString, Float64}(all_row_labels .=> NaN)
        mle_errors["Time"] = sum(mle_sim_results["solvetime"]) / (60 * 36) # convert to min and divide by total threads
        for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θmle_derived)
            mle_errors[lab] = mean(abs, filter(!isnan, θt .- θi))
        end
        push!(all_errors, "MLE" => mle_errors)

        for mode in [:mean], maxiter in [1,2,5]
            maxiter_lab = "$maxiter sample" * ifelse(maxiter > 1, "s", "")
            @info "Compute CVAE inference error ($maxiter_lab)..."
            cvae_errors = Dict{AbstractString, Float64}(all_row_labels .=> NaN)
            cvae_errors["Time"]  = @elapsed cvae_state = inverter(X̂true |> to32; maxiter, mode)
            cvae_errors["Time"] /= 60 # convert sec => min
            θcvae_derived = cvae_state.θ |> θderived_cpu
            for (lab, θt, θi) in zip(θderivedlabels(phys), θtrue_derived, θcvae_derived)
                cvae_errors[lab] = mean(abs, filter(!isnan, θt .- θi))
            end
            push!(all_errors, "CVAE ($maxiter_lab)" => cvae_errors)
        end

        label_s_to_ms(unit) = ifelse(unit == "s", "ms", unit)
        value_s_to_ms(val, unit) = ifelse(unit == "s", 1000*val, val)
        table_header = [name for (name, _) in all_errors]
        table_row_names = all_row_labels .* " [" .* label_s_to_ms.(all_row_units) .* "]"
        table_data = [value_s_to_ms(err[row], unit) for (row, unit) in zip(all_row_labels, all_row_units), (_, err) in all_errors]

        default_pretty_table(stdout, table_data, table_header, table_row_names; backend = :text)
        for (backend, filename) in [(:text, "errors.txt"), (:latex, "errors.tex")]
            open(io -> default_pretty_table(io, table_data, table_header, table_row_names; backend), joinpath(savefolder, filename); write = true)
        end
    end
end

function default_pretty_table(io, data, header, row_names; backend = :text, kwargs...)
    is_minimum = (data,i,j) -> !isnan(data[i,j]) && data[i,j] ≈ minimum(filter(!isnan, data[i,:]))
    hl = if backend === :text
        PrettyTables.Highlighter(is_minimum, foreground = :blue, bold = true)
    else
        PrettyTables.LatexHighlighter(is_minimum, ["color{blue}", "textbf"])
    end
    PrettyTables.pretty_table(io, data, header; backend, row_names, highlighters = (hl,), formatters = (v,i,j) -> round(v, sigdigits = 3), body_hlines = [size(data,1)-1], kwargs...)
end

####
#### Peak separation
####

function peak_separation(
        phys::BiexpEPGModel,
        models,
        derived,
        img::CPMGImage;
        cvae_iters = 10,
        savefolder = nothing,
        savetypes = [".png"],
    )

    nT2, nSNR = 100, 100
    settings = let
        T2ratio  = repeat(range(1.5, 4.0; length = nT2), 1, nSNR)
        T2mean   = fill(50e-3, nT2, nSNR)
        SNR      = repeat(range(40.0, 100.0; length = nSNR) |> permutedims, nT2, 1)
        epsilon  = 10.0.^(.-SNR./20)
        alpha    = 150.0 .+ 30 .* rand(nT2, nSNR)
        refcon   = fill(180.0, nT2, nSNR) # 150.0 .+ 30 .* rand(nT2, nSNR)
        T2short  = T2mean ./ sqrt.(T2ratio)
        T2long   = T2mean .* sqrt.(T2ratio)
        Ashort   = 0.1 .+ 0.3 .* rand(nT2, nSNR)
        Along    = 1.0 .- Ashort
        T1       = fill(Float64(T1time(img)), nT2, nSNR)
        TE       = fill(Float64(echotime(img)), nT2, nSNR)
        (; T2ratio, T2mean, SNR, epsilon, alpha, refcon, T2short, T2long, Ashort, Along, T1, TE)
    end

    args = vec.((settings.alpha, settings.refcon, settings.T2short, settings.T2long, settings.Ashort, settings.Along, settings.T1, settings.TE))
    X = _signal_model_f64(phys, args...)[1:nsignal(img), :]
    X = X ./ maximum(X; dims = 1)
    X̂ = add_noise_instance(phys, X, vec(settings.epsilon)')
    X̂meta = MetaCPMGSignal(phys, img, X̂ |> to32)

    function cvae_inference_state(Ymeta)
        state = posterior_state(
            phys,
            models["genatr"],
            derived["cvae"],
            Ymeta;
            miniter = cvae_iters,
            maxiter = cvae_iters,
            alpha   = 0.0,
            verbose = false,
            mode    = :maxlikelihood,
        )
        (; θ = θderived(phys, img, state.θ))
    end

    function decaes_inference_state(Ymeta)
        Ydecaes = permutedims(reshape(signal(Ymeta), size(signal(Ymeta))..., 1, 1), (2,3,4,1)) # nTE x nbatch -> nbatch x 1 x 1 x nTE
        t2mapopts = DECAES.T2mapOptions(img.t2mapopts, MatrixSize = size(Ydecaes)[1:3])
        t2partopts = DECAES.T2partOptions(img.t2partopts, MatrixSize = size(Ydecaes)[1:3])
        t2maps, t2dist = DECAES.T2mapSEcorr(Ydecaes |> arr64, t2mapopts)
        t2parts = DECAES.T2partSEcorr(t2dist, t2partopts) # size(t2dist) = nbatch x 1 x 1 x nT2
        (; t2maps, t2dist, t2parts)
    end

    cvae_results = let
        @unpack θ = cvae_inference_state(X̂meta)
        (; T2short = arr64(reshape(θ.T2short, nT2, nSNR)), T2long = arr64(reshape(θ.T2long, nT2, nSNR)))
    end

    #### T2 peaks plots

    decaes_results = let
        @unpack t2maps, t2dist, t2parts = decaes_inference_state(X̂meta)
        T2short, T2long = zeros(nT2, nSNR), zeros(nT2, nSNR)
        # iperm = zeros(Int, size(t2dist, 4))
        # for i in 1:size(t2dist, 1)
        #     sortperm!(iperm, view(t2dist, i, 1, 1, :); rev = true)
        #     T21 = t2maps["t2times"][iperm[1]]
        #     T22 = t2maps["t2times"][iperm[2]]
        #     T2short[i], T2long[i] = min(T21, T22), max(T21, T22)
        # end
        icutoff = findlast(t2maps["t2times"] .<= settings.T2mean[1])
        for i in 1:size(t2dist, 1)
            T2short[i] = t2maps["t2times"][findmax(@views t2dist[i,1,1,1:icutoff])[2]]
            T2long[i]  = t2maps["t2times"][icutoff + findmax(@views t2dist[i,1,1,icutoff+1:end])[2]]
        end
        (; T2short, T2long)
    end

    for T2field in (:T2short, :T2long), (name, results) in [(:cvae, cvae_results), (:decaes, decaes_results)]
        err = 1000 .* abs.(getfield(settings, T2field) .- getfield(results, T2field))
        filename = joinpath(mkpath(savefolder), "t2peaks-$name-$T2field")
        pyheatmap(
            err;
            title = "$T2field error vs. $T2field and SNR",
            clim = (0.0, 15.0),
            aspect = 1.0,
            extent = [settings.SNR[1,[1,end]]..., 1000 .* getfield(settings, T2field)[[end,1],1]...], # [left, right, bottom, top]
            xlabel = "SNR",
            ylabel = "$T2field = T2mean $(T2field === :T2long ? "*" : "/") T2ratio [ms]",
            filename,
            savetypes,
        )
    end

    # T2 distribution plots
    let
        i, j = 30, 50
        X̂meta = MetaCPMGSignal(phys, img, repeat(reshape(X̂, :, nT2, nSNR)[:,i,j], 1, 1000) |> to32)
        cvae_θ = cvae_inference_state(X̂meta).θ
        @unpack t2maps, t2dist, t2parts = decaes_inference_state(X̂meta[:,1:1])

        saveplot(p, name) = map(suf -> savefig(p, joinpath(mkpath(savefolder), name * suf)), savetypes)
        T2short_lab, T2long_lab, SNR_lab = map(x->round(x;digits=1), (1000 * settings.T2short[i,j], 1000 * settings.T2long[i,j], settings.SNR[i,j]))
        let
            T2 = 1000 .* vcat(cvae_θ.T2short, cvae_θ.T2long) |> vec |> arr64
            A = vcat(cvae_θ.Ashort, cvae_θ.Along) |> vec |> arr64
            p = sticks(
                T2, A;
                label = L"$T_2$ Distribution", ylabel = L"$T_2$ Amplitude [a.u.]", xlabel = L"$T_2$ [ms]",
                title = L"$T_{2,short}$ = %$(T2short_lab) ms, $T_{2,long}$ = %$(T2long_lab) ms, SNR = %$(SNR_lab)",
                # xscale = :log10, xlim = 1000 .* (10e-3, 100e-3), xticks = 10 .^ (1.0:0.25:2.0),
                xscale = :identity, xlim = 1000 .* (10e-3, 100e-3), xticks = 10:10:100,
                titlefontsize = 14, labelfontsize = 12, xtickfontsize = 12, ytickfontsize = 12, legendfontsize = 10, #TODO
                marker = (:black, :circle, 1),
                formatter = x -> round(x, digits = 1),
            )
            vline!(p, 1000 .* [settings.T2short[i,j], settings.T2long[i,j]]; label = L"True $T_2$", lw = 3)
            vline!(p, 1000 .* [mean(cvae_θ.T2short), mean(cvae_θ.T2long)]; label = L"Recovered $T_2$", lw = 3)
            saveplot(p, "t2peaks-cvae-samples")
        end
        let
            p = plot(
                1000 .* t2maps["t2times"], t2dist[1,1,1,:];
                label = L"$T_2$ Distribution", ylabel = L"$T_2$ Amplitude [a.u.]", xlabel = L"$T_2$ [ms]",
                title = L"$T_{2,short}$ = %$(T2short_lab) ms, $T_{2,long}$ = %$(T2long_lab) ms, SNR = %$(SNR_lab)",
                # xscale = :log10, xlim = 1000 .* (10e-3, 100e-3), xticks = 10 .^ (1.0:0.25:2.0),
                xscale = :identity, xlim = 1000 .* (10e-3, 100e-3), xticks = 10:10:100,
                titlefontsize = 14, labelfontsize = 12, xtickfontsize = 12, ytickfontsize = 12, legendfontsize = 10, #TODO
                marker = (:black, :circle, 3), line = (:blue, 3),
                formatter = x -> round(x, digits = 1),
            )
            vline!(p, 1000 .* [settings.T2short[i,j], settings.T2long[i,j]]; label = L"True $T_2$", lw = 3)
            vline!(p, 1000 .* [decaes_results.T2short[i,j], decaes_results.T2long[i,j]]; label = L"Recovered $T_2$", lw = 3)
            saveplot(p, "t2peaks-decaes-samples")
        end
    end

    return (; settings, cvae_results, decaes_results)
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
