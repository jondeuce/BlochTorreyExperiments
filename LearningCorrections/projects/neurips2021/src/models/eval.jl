####
#### Abstract type dictionary for storing arbitrary caches by element type.
#### Definition taken directly from the folks at RigidBodyDynamics.jl:
####
####    https://github.com/JuliaRobotics/RigidBodyDynamics.jl/blob/b9ef1d6974beff4d4fbe7dffc6dbfa65f71e0132/src/caches.jl#L1
####
#### Following discussions re: pre-allocating caches for Dual number types:
####
####    https://discourse.julialang.org/t/nlsolve-preallocated-forwarddiff-dual/13950
####    https://github.com/JuliaRobotics/RigidBodyDynamics.jl/issues/548
####    https://diffeq.sciml.ai/stable/basics/faq/#I-get-Dual-number-errors-when-I-solve-my-ODE-with-Rosenbrock-or-SDIRK-methods

abstract type AbstractTypeDict end
function valuetype end
function makevalue end

function Base.getindex(c::C, ::Type{T}) where {C<:AbstractTypeDict, T}
    ReturnType = valuetype(C, T)
    key = objectid(T)
    @inbounds for i in eachindex(c.keys)
        if c.keys[i] === key
            return c.values[i]::ReturnType
        end
    end
    value = makevalue(c, T)::ReturnType
    push!(c.keys, key)
    push!(c.values, value)
    value::ReturnType
end

#### Array caches

struct ArrayCache{N,Size} <: AbstractTypeDict
    keys::Vector{UInt}
    values::Vector{Array}
end
ArrayCache(Size::Int...) = ArrayCache(Size)
ArrayCache(Size::NTuple{N,Int}) where {N} = ArrayCache{N,Size}([], [])
@inline valuetype(::Type{ArrayCache{N,Size}}, ::Type{T}) where {T, N, Size} = Array{T,N}
@inline makevalue(::ArrayCache{N,Size}, ::Type{T}) where {T, N, Size} = zeros(T, Size)

#### BiexpEPGModel caches

struct EPGModelWorkCache{ETL} <: AbstractTypeDict
    keys::Vector{UInt}
    values::Vector{DECAES.EPGWork_ReIm_DualMVector_Split}
end
EPGModelWorkCache(ETL::Int) = EPGModelWorkCache{ETL}([], [])
@inline valuetype(::Type{EPGModelWorkCache{ETL}}, ::Type{T}) where {T, ETL} = DECAES.EPGWork_ReIm_DualMVector_Split{T, ETL, DECAES.MVector{ETL,DECAES.SVector{3,T}}, DECAES.MVector{ETL,T}}
@inline makevalue(::EPGModelWorkCache{ETL}, ::Type{T}) where {T, ETL} = DECAES.EPGWork_ReIm_DualMVector_Split(T, ETL)

#### BiexpEPGModel caches

struct BiexpEPGModelWorkCache{ETL} <: AbstractTypeDict
    keys::Vector{UInt}
    values::Vector{BiexpEPGModelWork}
end
BiexpEPGModelWorkCache(ETL::Int) = BiexpEPGModelWorkCache{ETL}([], [])
@inline valuetype(::Type{BiexpEPGModelWorkCache{ETL}}, ::Type{T}) where {T, ETL} = BiexpEPGModelWork{T, ETL, DECAES.MVector{ETL,T}, valuetype(EPGModelWorkCache{ETL}, T), valuetype(EPGModelWorkCache{ETL}, T)}
@inline makevalue(::BiexpEPGModelWorkCache{ETL}, ::Type{T}) where {T, ETL} = BiexpEPGModelWork(T, ETL)

####
#### MCMC inference
####

function mcmc_biexp_epg_work_factory(phys, img)
    @unpack TEbd, T2bd, T1bd = phys
    TE        = 1.0 # all times are relative to TE and unitless
    T1        = T1time(img) / echotime(img) |> Float64
    logτ2lo   = log(T2bd[1] / TEbd[2]) |> Float64
    logτ2hi   = log(T2bd[2] / TEbd[1]) |> Float64
    logτ1lo   = log(T1bd[1] / TEbd[2]) |> Float64
    logτ1hi   = log(T1bd[2] / TEbd[1]) |> Float64
    logτ2lo′  = log(T2bd[1] / echotime(img)) |> Float64
    logτ2hi′  = log(T2bd[2] / echotime(img)) |> Float64
    δ0        = (log(T1) - logτ1lo) / (logτ1hi - logτ1lo)
    epg_cache = BiexpEPGModelWorkCache(nsignal(img))
    return (; TEbd, T2bd, T1bd, TE, T1, logτ2lo, logτ2hi, logτ1lo, logτ1hi, logτ2lo′, logτ2hi′, δ0, epg_cache)
end

function biexp_epg_model_scaled!(work, α, β, η, δ1, δ2, logϵ, logs, ::Val{normalized} = Val(true), ::Val{scaled} = Val(true)) where {normalized, scaled}
    @unpack TE, T1, logτ2lo, logτ2hi, epg_cache = work
    D      = promote_type(typeof(α), typeof(β), typeof(η), typeof(δ1), typeof(δ2), typeof(logϵ), typeof(logs))
    epg    = epg_cache[D]
    T21    = exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
    T22    = exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
    A1, A2 = η, 1-η
    ϵ, s   = exp(logϵ), exp(logs)
    ψ      = (α, β, T21, T22, A1, A2, T1, TE)
    X      = _biexp_epg_model_f64!(epg.dc, epg, ψ)

    if normalized
        Xmax = zero(eltype(X))
        @simd for i in eachindex(X)
            Xmax = max(X[i], Xmax)
        end
    else
        Xmax = one(eltype(X))
    end

    if scaled
        Xscale = s / Xmax
    else
        Xscale = inv(Xmax)
    end

    if normalized || scaled
        @simd for i in eachindex(X)
            X[i] *= Xscale
        end
    end

    return X
end

Turing.@model function mcmc_biexp_epg_model!(work, Y::AbstractVector{Float64})
    α     ~ Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0)
    β     ~ Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0)
    η     ~ Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0)
    δ1    ~ Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0)
    δ2    ~ Distributions.TruncatedNormal(1.0, 0.5, 0.0, 1.0)
    logϵ  ~ Distributions.Uniform(log(1e-5), log(1e-1)) # equivalent to 20 <= SNR <= 100
    logs  ~ Distributions.TruncatedNormal(0.0, 0.5, -2.5, 2.5)
    X     = biexp_epg_model_scaled!(work, α, β, η, δ1, δ2, logϵ, logs)
    logϵs = logϵ + logs
    # ϵs  = exp(logϵs)
    for i in 1:length(Y)
        logℒ = -neglogL_rician(Y[i], X[i], logϵs)
        Turing.@addlogprob! logℒ
        # Y[i] ~ Rician(X[i], ϵs)
    end
end

function mcmc_biexp_epg(
        phys;
        img_idx         = 1,
        num_samples     = 100,
        dataset         = :val,
        total_chains    = Colon(),
        seed            = 0,
        shuffle         = total_chains !== Colon(),
        save            = true,
        savefolder      = "",
        checkpoint      = save,
        checkpoint_freq = 2048, # checkpoint every `checkpoint_freq` iterations
        progress_freq   = 15.0, # update progress bar every `progress_freq` seconds
    )

    file_prefix(ischeckpoint) = joinpath(savefolder, (ischeckpoint ? "checkpoint_" : "") * "image-$(img_idx)_dataset-$(dataset)")
    file_suffix(chains) = "$(lpad(chains[1].chain_idx, 7, '0'))-to-$(lpad(chains[end].chain_idx, 7, '0'))"

    function save_chains(chains; ischeckpoint)
        filename = file_prefix(ischeckpoint) * "_" * file_suffix(chains) * ".jld2"
        FileIO.save(filename, Dict("chains" => chains))
    end

    function save_dataframe(chains; ischeckpoint)
        file_basename = file_prefix(ischeckpoint) * "_" * file_suffix(chains)
        chains_df = mapreduce(vcat, chains) do (chain_idx, img_col, img_xyz, chain)
            df = DataFrame(chain)
            df.dataset_col = fill(img_col, DataFrames.nrow(df))
            df.image_x = fill(img_xyz[1], DataFrames.nrow(df))
            df.image_y = fill(img_xyz[2], DataFrames.nrow(df))
            df.image_z = fill(img_xyz[3], DataFrames.nrow(df))
            return df
        end
        # CSV.write(file_basename * ".csv", chains_df)

        # Rename columns and save to .mat (Matlab variables cannot contain unicode chars)
        df_names   = [:iteration, :image_x, :image_y, :image_z, :α, :β, :η, :δ1, :δ2, :logϵ, :logs]
        mat_names  = ["iteration", "image_x", "image_y", "image_z", "alpha", "beta", "eta", "delta1", "delta2", "logepsilon", "logscale"]
        mat_data   = Dict([k_mat => copy(getproperty(chains_df, k_df)) for (k_df, k_mat) in zip(df_names, mat_names)])
        DECAES.MAT.matwrite(file_basename * ".mat", mat_data)
    end

    img              = phys.images[img_idx]
    ydata            = img.partitions[dataset] .|> Float64
    yindices         = img.indices[dataset]
    work_buffers     = [mcmc_biexp_epg_work_factory(phys, img) for _ in 1:Threads.nthreads()]

    total_chains     = total_chains == Colon() ? size(ydata, 2) : total_chains
    img_cols         = !shuffle ? (1:total_chains) : sample(MersenneTwister(seed), 1:size(ydata, 2), total_chains; replace = false)
    chains           = Any[nothing for _ in 1:total_chains]
    checkpoint_cb    = !(save && checkpoint) ? nothing : Js -> save_dataframe(view(chains, Js); ischeckpoint = true)

    Turing.setprogress!(false)
    Turing.setchunksize(7)

    foreach_with_progress(eachindex(img_cols); checkpoint_cb, checkpoint_freq, dt = progress_freq) do Jc
        J     = img_cols[Jc]
        work  = work_buffers[Threads.threadid()]
        Y     = ydata[:,J]
        model = mcmc_biexp_epg_model!(work, Y)
        chain = DECAES.tee_capture(suppress_terminal = true, suppress_logfile = true) do io
            Turing.sample(model, Turing.NUTS(0.65), num_samples)
        end
        chains[Jc] = (; chain_idx = Jc, img_col = J, img_xyz = yindices[J], chain = chain)
    end

    save && (@info "Saving chains to .mat file..."; @time save_dataframe(chains; ischeckpoint = false))
    # save && (@info "Saving chains to .jld2 file..."; @time save_chains(chains; ischeckpoint = false))

    return chains
end

####
#### MLE inference
####

function mle_biexp_epg_noise_only(
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
        initial_logϵ  = nothing,
        initial_logs  = nothing;
        freeze_logϵ   = false,
        freeze_logs   = false,
        logϵ_bounds   = nothing,
        logs_bounds   = nothing,
        batch_size    = 128 * Threads.nthreads(),
        verbose       = true,
        dryrun        = false,
        dryrunsamples = batch_size,
        dryrunshuffle = true,
        dryrunseed    = 0,
        opt_alg       = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args      = Dict{Symbol,Any}(),
    )
    @assert size(X) == size(Y)
    @assert !(freeze_logϵ && freeze_logs) # must be optimizing at least one

    dryrun && let
        I, _ = sample_maybeshuffle(1:size(Y,2); shuffle = dryrunshuffle, samples = dryrunsamples, seed = dryrunseed)
        X, Y = X[:,I], Y[:,I]
        (initial_logϵ !== nothing) && (initial_logϵ = initial_logϵ[:,I])
        (initial_logs !== nothing) && (initial_logs = initial_logs[:,I])
    end

    # Initial guess
    (initial_logϵ === nothing) && (initial_logϵ = log.(sqrt.(mean(abs2, X .- Y; dims = 1))))
    (initial_logs === nothing) && (initial_logs = .-log.(maximum(mean_rician.(X, exp.(initial_logϵ)); dims = 1)))
    initial_loss  = sum(neglogL_rician.(Y, exp.(initial_logs) .* X, initial_logs .+ initial_logϵ); dims = 1)
    initial_guess = (
        logϵ = initial_logϵ |> cpu64,
        logs = initial_logs |> cpu64,
        ℓ    = initial_loss |> cpu64,
    )
    X, Y = (X, Y) .|> cpu64

    # Setup
    noptvars     = !freeze_logϵ + !freeze_logs
    logϵ_bounds  = logϵ_bounds === nothing ? (-Inf, Inf) : logϵ_bounds
    logs_bounds  = logs_bounds === nothing ? (-Inf, Inf) : logs_bounds
    lower_bounds = Float64[]; !freeze_logϵ && push!(lower_bounds, logϵ_bounds[1]); !freeze_logs && push!(lower_bounds, logs_bounds[1])
    upper_bounds = Float64[]; !freeze_logϵ && push!(upper_bounds, logϵ_bounds[2]); !freeze_logs && push!(upper_bounds, logs_bounds[2])

    work_spaces = map(1:Threads.nthreads()) do _
        (
            Xj    = zeros(size(X,1)),
            Yj    = zeros(size(Y,1)),
            x0    = zeros(noptvars),
            logϵ0 = Ref(0.0),
            logs0 = Ref(0.0),
            ∇res  = ForwardDiff.DiffResults.GradientResult(zeros(noptvars)),
            ∇cfg  = ForwardDiff.GradientConfig(nothing, zeros(noptvars), ForwardDiff.Chunk(noptvars)),
            opt   = let
                opt = NLopt.Opt(opt_alg, noptvars)
                opt.xtol_rel      = 1e-8
                opt.ftol_rel      = 1e-8
                opt.maxeval       = 250
                opt.maxtime       = 1.0
                opt.lower_bounds  = lower_bounds
                opt.upper_bounds  = upper_bounds
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

    function f(work, x::Vector{D}) where {D <: MaybeDualF64}
        @inbounds begin
            logϵ  = freeze_logϵ ? work.logϵ0[] : freeze_logs ? x[1] : x[1]
            logs  = freeze_logs ? work.logs0[] : freeze_logϵ ? x[1] : x[2]
            s     = exp(logs)
            logϵs = logs + logϵ # log(s*ϵ)
            ℓ     = zero(D)
            @simd for i in eachindex(work.Xj)
                yi = work.Yj[i]
                νi = s * work.Xj[i]
                ℓ += neglogL_rician(yi, νi, logϵs) # Rician negative log likelihood
            end
            return ℓ
        end
    end

    function fg!(work, x::Vector{Float64}, g::Vector{Float64})
        if length(g) > 0
            # simple_fd_gradient!(g, _x -> f(work, _x), x)
            ForwardDiff.gradient!(work.∇res, _x -> f(work, _x), x, work.∇cfg)
            y  = ForwardDiff.DiffResults.value(work.∇res)
            ∇y = ForwardDiff.DiffResults.gradient(work.∇res)
            @avx g .= ∇y
            return y
        else
            f(work, x)
        end
    end

    start_time = time()
    batches = batch_size === Colon() ? [1:size(Y,2)] : Iterators.partition(1:size(Y,2), batch_size)
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Xj               .= X[:,j]
                work.Yj               .= Y[:,j]
                work.logϵ0[]           = initial_guess.logϵ[1,j]
                work.logs0[]           = initial_guess.logs[1,j]
                work.x0               .= freeze_logϵ ? work.logs0[] : freeze_logs ? work.logϵ0[] : (work.logϵ0[], work.logs0[])
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.logepsilon[j]  = freeze_logϵ ? work.logϵ0[] : freeze_logs ? minx[1] : minx[1]
                results.logscale[j]    = freeze_logs ? work.logs0[] : freeze_logϵ ? minx[1] : minx[2]
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
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))"
    end

    BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end

function mle_biexp_epg_θ_to_x(work::W, θ::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack logτ2lo, logτ2hi, logτ2lo′, logτ2hi′ = work
        α, β, η, δ1, δ2, logϵ, logs = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7]
        t   = 1e-12
        δ1′ = clamp(logτ2lo + (logτ2hi - logτ2lo) * δ1, logτ2lo′ + t, logτ2hi′ - t)
        δ2′ = clamp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)), logτ2lo′ + t, logτ2hi′ - t)
        z1  = (δ1′ - logτ2lo′) / (logτ2hi′ - logτ2lo′)
        z2  = ((δ2′ - logτ2lo′) / (logτ2hi′ - logτ2lo′) - z1) / (1 - z1)
        return α, β, η, z1, z2, logϵ, logs
    end
end

function mle_biexp_epg_x_to_θ(work::W, x::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack logτ2lo, logτ2hi, logτ2lo′, logτ2hi′, δ0 = work
        α, β, η, z1, z2, logϵ, logs = x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        δ1 = ((logτ2lo′ - logτ2lo) + (logτ2hi′ - logτ2lo′) * z1) / (logτ2hi - logτ2lo)
        δ2 = (((logτ2lo′ - logτ2lo) + (logτ2hi′ - logτ2lo′) * (z1 + z2 * (1 - z1))) / (logτ2hi - logτ2lo) - δ1) / (1 - δ1)
        return α, β, η, δ1, δ2, logϵ, logs
    end
end

function mle_biexp_epg_θ_to_ψ(work::W, θ::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack logτ2lo, logτ2hi, logτ1lo, logτ1hi, δ0 = work
        α, β, η, δ1, δ2, logϵ, logs = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7]
        T21 = exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
        T22 = exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
        A21 = η
        A22 = 1 - η
        T1 = exp(logτ1lo + (logτ1hi - logτ1lo) * δ0)
        return α, β, T21, T22, A21, A22, T1, one(T1)
    end
end

function mle_biexp_epg_x_regularization(work::W, x::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack xlo, xhi, σreg = work
        α,   β,   η,   z1,   z2,   logϵ,   logs   =   x[1],   x[2],   x[3],   x[4],   x[5],   x[6],   x[7] # flip/refcon angles, mwf, and relative log T2 short/long
        αlo, βlo, ηlo, z1lo, z2lo, logϵlo, logslo = xlo[1], xlo[2], xlo[3], xlo[4], xlo[5], xlo[6], xlo[7] # parameter lower bounds
        αhi, βhi, ηhi, z1hi, z2hi, logϵhi, logshi = xhi[1], xhi[2], xhi[3], xhi[4], xhi[5], xhi[6], xhi[7] # parameter upper bounds
        # Regularization is negative log-likelihood of a sum of Gaussian distributions centred on parameter interval lower/upper bounds.
        # This biases parameters to prefer to be at one endpoint or the other in cases where the solution is ill-posed.
        # Gaussian widths are equal to σreg * (interval width). Normalization constant is discarded.
        R = (
            ((α - αhi) / (αhi - αlo))^2 + # flip angle biased toward 180 deg
            ((β - βhi) / (βhi - βlo))^2 + # refcon angle biased toward 180 deg
            ((η - ηlo) / (ηhi - ηlo))^2 + # mwf biased toward 0%
            ((z1 - z1lo) / (z1hi - z1lo))^2 + # (relative-)log T2 short biased lower
            ((z2 - z2hi) / (z2hi - z2lo))^2 +  # (relative-)log T2 long biased higher
            logs^2 # scale biased to be near zero
        ) / (2 * σreg^2) # regularization strength
        return R
    end
end

@inline mle_biexp_epg_work(work::W, ::Type{Float64}) where {W} = work.epg
@inline mle_biexp_epg_work(work::W, ::Type{D}) where {W, D <: Dual64} = work.∇epg

function f_mle_biexp_epg!(work::W, x::Vector{D}, ::Val{verbose} = Val(false)) where {W, D <: MaybeDualF64, verbose}
    @inbounds begin
        θ      = mle_biexp_epg_x_to_θ(work, x) # θ = α, β, η, z1, z2, logϵ, logs
        logϵ   = θ[end-1]
        logs   = θ[end]
        s      = exp(logs)
        ψ      = mle_biexp_epg_θ_to_ψ(work, θ) # ψ = alpha, refcon, T2short, T2long, Ashort, Along, T1, TE
        epg    = mle_biexp_epg_work(work, D)
        X      = _biexp_epg_model_f64!(epg.dc, epg, ψ)
        Xmax   = zero(D)
        @simd for i in eachindex(X)
            Xmax = max(X[i], Xmax)
        end
        Xscale = s / Xmax # normalize X to maximum 1 and scale by s; hoist outside loop
        logϵi  = logs + logϵ # hoist outside loop
        ℓ      = zero(D)
        @simd for i in eachindex(work.Y)
            X[i] *= Xscale
            ℓ    += neglogL_rician(work.Y[i], X[i], logϵi) # Rician negative log likelihood
        end
        R      = mle_biexp_epg_x_regularization(work, x)
        work.neglogL[] = ForwardDiff.value(ℓ)
        work.reg[] = ForwardDiff.value(R)
        return ℓ + R
    end
end

function fg_mle_biexp_epg!(work::W, x::Vector{Float64}, g::Vector{Float64}) where {W}
    if length(g) > 0
        ForwardDiff.gradient!(work.∇res, _x -> f_mle_biexp_epg!(work, _x), x, work.∇cfg)
        y  = ForwardDiff.DiffResults.value(work.∇res)
        ∇y = ForwardDiff.DiffResults.gradient(work.∇res)
        @avx g .= ∇y
        return y
    else
        return f_mle_biexp_epg!(work, x)
    end
end

function mle_biexp_epg_initial_guess(
        phys::BiexpEPGModel,
        img::CPMGImage,
        cvae::Union{Nothing,<:CVAE} = nothing;
        data_subset        = :mask,  # One of :mask, :train, :val, :test
        refine_init_logϵ   = false,
        refine_init_logs   = false,
        gpu_batch_size     = 2048,
        verbose            = true,
        dryrun             = false,
        dryrunsamples      = gpu_batch_size,
        dryrunshuffle      = true,
        dryrunseed         = 0,
    )
    @assert data_subset ∈ (:mask, :train, :val, :test)

    # MLE for whole image of simulated data
    image_indices = img.indices[data_subset]
    if dryrun
        image_indices, _ = sample_maybeshuffle(image_indices; shuffle = dryrunshuffle, samples = dryrunsamples, seed = dryrunseed)
    end
    image_data = img.data[image_indices, :] |> gpu |> permutedims

    Ymeta = MetaCPMGSignal(phys, img, image_data)
    batches = gpu_batch_size === Colon() ? [1:size(signal(Ymeta),2)] : Iterators.partition(1:size(signal(Ymeta),2), gpu_batch_size)
    initial_guess = mapreduce((x,y) -> map(hcat, x, y), batches) do batch
        if cvae === nothing
            θ = sampleθprior(phys, typeof(signal(Ymeta)), length(batch))
            Z = zeros_similar(θ, 0, size(θ,2))
            state = posterior_state(phys, Ymeta[:,batch], θ, Z)
        else
            state = posterior_state(phys, cvae, Ymeta[:,batch])
        end
        map(cpu64, state)
    end

    if refine_init_logϵ || refine_init_logs
        _, refined_results = mle_biexp_epg_noise_only(
            initial_guess.X |> cpu64,
            signal(Ymeta)   |> cpu64,
            refine_init_logϵ ? initial_guess.θ[6:6,:] |> cpu64 : nothing,
            refine_init_logs ? initial_guess.θ[7:7,:] |> cpu64 : nothing;
            freeze_logϵ = !refine_init_logϵ,
            freeze_logs = !refine_init_logs,
            logϵ_bounds = θbounds(phys)[6],
            logs_bounds = θbounds(phys)[7],
            verbose     = verbose,
        )
        refine_init_logϵ && (initial_guess.θ[6:6,:] .= arr_similar(initial_guess.θ, refined_results.logepsilon'))
        refine_init_logs && (initial_guess.θ[7:7,:] .= arr_similar(initial_guess.θ, refined_results.logscale'))
    end

    return initial_guess
end

function mle_biexp_epg(
        phys::BiexpEPGModel,
        img::CPMGImage,
        cvae::Union{Nothing,<:CVAE} = nothing;
        batch_size         = 128 * Threads.nthreads(),
        sigma_reg          = 0.5,
        verbose            = true,
        initial_guess      = nothing,
        initial_guess_args = Dict{Symbol,Any}(
            :refine_init_logϵ => true,
            :refine_init_logs => true,
            :verbose          => verbose,
        ),
        opt_alg            = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args           = Dict{Symbol,Any}(),
    )

    # Compute initial guess if not given
    if initial_guess === nothing
        initial_guess = mle_biexp_epg_initial_guess(phys, img, cvae; initial_guess_args...)
    end

    num_optvars   = nmarginalized(phys)
    lower_bounds  = θmarginalized(phys, θlower(phys)) |> cpu64
    upper_bounds  = θmarginalized(phys, θupper(phys)) |> cpu64

    work_spaces = map(1:Threads.nthreads()) do _
        @unpack TEbd, T2bd, T1bd = phys
        (
            neglogL  = Ref(0.0),
            reg      = Ref(0.0),
            σreg     = sigma_reg |> Float64,
            Y        = zeros(nsignal(img)),
            x0       = zeros(num_optvars),
            xlo      = copy(lower_bounds),
            xhi      = copy(upper_bounds),
            dx       = copy(upper_bounds .- lower_bounds),
            logτ2lo  = log(T2bd[1] / TEbd[2]) |> Float64,
            logτ2hi  = log(T2bd[2] / TEbd[1]) |> Float64,
            logτ1lo  = log(T1bd[1] / TEbd[2]) |> Float64,
            logτ1hi  = log(T1bd[2] / TEbd[1]) |> Float64,
            logτ2lo′ = log(T2bd[1] / echotime(img)) |> Float64,
            logτ2hi′ = log(T2bd[2] / echotime(img)) |> Float64,
            δ0       = θnuissance(phys, img) |> Float64,
            epg      = BiexpEPGModelWork(Float64, nsignal(img)),
            ∇epg     = BiexpEPGModelWork(ForwardDiff.Dual{Nothing, Float64, num_optvars}, nsignal(img)),
            ∇res     = ForwardDiff.DiffResults.GradientResult(zeros(num_optvars)),
            ∇cfg     = ForwardDiff.GradientConfig(nothing, zeros(num_optvars), ForwardDiff.Chunk(num_optvars)),
            opt      = let
                opt = NLopt.Opt(opt_alg, num_optvars)
                opt.lower_bounds  = copy(lower_bounds)
                opt.upper_bounds  = copy(upper_bounds)
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

    num_signals, num_problems = size(initial_guess.Y)
    results = (
        signalfit  = fill(NaN, num_signals, num_problems),
        theta      = fill(NaN, ntheta(phys), num_problems),
        loss       = fill(NaN, num_problems),
        reg        = fill(NaN, num_problems),
        retcode    = fill(NaN, num_problems),
        numevals   = fill(NaN, num_problems),
        solvetime  = fill(NaN, num_problems),
    )

    start_time = time()
    batches = Iterators.partition(1:num_problems, batch_size === Colon() ? num_problems : batch_size)
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Y                 .= initial_guess.Y[:,j]
                work.x0                .= mle_biexp_epg_θ_to_x(work, view(initial_guess.θ, :, j))
                work.opt.min_objective  = (x, g) -> fg_mle_biexp_epg!(work, x, g)

                solvetime               = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                f_mle_biexp_epg!(work, minx) # update `work` with solution

                results.signalfit[:,j] .= work.epg.dc
                results.theta[:,j]     .= mle_biexp_epg_x_to_θ(work, minx)
                results.loss[j]         = work.neglogL[]
                results.reg[j]          = work.reg[]
                results.retcode[j]      = Base.eval(NLopt, ret) |> Int #TODO cleaner way to convert Symbol to enum?
                results.numevals[j]     = work.opt.numevals
                results.solvetime[j]    = solvetime
            end
        end

        # Checkpoint results
        elapsed_time   = time() - start_time
        remaining_time = (elapsed_time / batchnum) * (length(batches) - batchnum)
        mle_per_second = batch[end] / elapsed_time
        verbose && @info "$batchnum / $(length(batches))" *
            " -- batch: $(DECAES.pretty_time(batchtime))" *
            " -- elapsed: $(DECAES.pretty_time(elapsed_time))" *
            " -- remaining: $(DECAES.pretty_time(remaining_time))" *
            " -- rate: $(round(mle_per_second; digits = 2))Hz" *
            (!haskey(initial_guess, :ℓ) ? "" : " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))") *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))" *
            " -- reg: $(round(mean(results.reg[1:batch[end]]); digits = 2))"
    end

    BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
end
