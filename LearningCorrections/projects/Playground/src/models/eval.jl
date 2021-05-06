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

#### EPGModel caches

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

function _test_epgwork_cache()
    T = Float64
    D = ForwardDiff.Dual{Nothing, Float64, 2}
    cache = ArrayCache(1,2,1)
    @btime $cache[$T]
    @btime $cache[$D]
    cache = EPGModelWorkCache(48)
    @btime $cache[$T]
    @btime $cache[$D]
    cache = BiexpEPGModelWorkCache(48)
    @btime $cache[$T]
    @btime $cache[$D]
end

#=
####
#### Recursively reinterpret arrays in nested structures.
#### Currently causes segfaults...
####

recurse_reinterpret(x, y::AbstractArray{T}) where {T} = recurse_reinterpret(x, T)
recurse_reinterpret(x, y::AbstractArray{V}) where {T, N, V <: DECAES.SVector{N,T}} = recurse_reinterpret(x, T)

recurse_reinterpret(x::AbstractArray{T1}, ::Type{T2}) where {T1, T2} = reinterpret(T2, x)
recurse_reinterpret(x::AbstractArray{V1}, ::Type{T2}) where {T1, T2, N, V1 <: DECAES.SVector{N,T1}} = reinterpret(DECAES.SVector{N,T2}, x)

function recurse_reinterpret(x, ::Type{T}) where {T}
    func, re = Flux.functor(x)
    return re(map(y -> recurse_reinterpret(y, T), func))
end

function Flux.functor(work::DECAES.EPGWork_ReIm_DualMVector_Split{<:Any, ETL}) where {ETL}
    function EPGWork_ReIm_DualMVector_Split_functor(
            MPSV₁::MPSVType,
            MPSV₂::MPSVType,
            decay_curve::DCType,
        ) where {T, MPSVType <: AbstractVector{DECAES.SVector{3,T}}, DCType <: AbstractVector{T}}
        return DECAES.EPGWork_ReIm_DualMVector_Split{T,ETL,MPSVType,DCType}(MPSV₁, MPSV₂, decay_curve)
    end
    return (work.MPSV₁, work.MPSV₂, work.decay_curve), caches -> EPGWork_ReIm_DualMVector_Split_functor(caches...)
end

function Flux.functor(work::BiexpEPGModelWork{<:Any, ETL}) where {ETL}
    function BiexpEPGModelWork_functor(
            dc::A,
            short_work::W1,
            long_work::W2,
        ) where {T, A <: AbstractVector{T}, W1 <: DECAES.AbstractEPGWorkspace{T}, W2 <: DECAES.AbstractEPGWorkspace{T}}
        return BiexpEPGModelWork{T,ETL,A,W1,W2}(dc, short_work, long_work)
    end
    return (work.dc, work.short_work, work.long_work), caches -> BiexpEPGModelWork_functor(caches...)
end

function _test_recurse_reinterp(phys)
    D1 = ForwardDiff.Dual{Nothing, Float64, 2}
    D2 = ForwardDiff.Dual{typeof(sin), Float64, 2}
    work1 = BiexpEPGModelWork(phys, D1, EPGVectorWorkFactory)
    @show typeof(work1)
    work2 = recurse_reinterpret(work1, D2)
    @show typeof(work2)
    return work1
end
=#

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

function biexp_epg_model_scaled!(work, α, β, η, δ1, δ2, logϵ, logs)
    @unpack TE, T1, logτ2lo, logτ2hi, epg_cache = work
    D      = promote_type(typeof(α), typeof(β), typeof(η), typeof(δ1), typeof(δ2), typeof(logϵ), typeof(logs))
    epg    = epg_cache[D]
    T21    = exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
    T22    = exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
    A1, A2 = η, 1-η
    ϵ, s   = exp(logϵ), exp(logs)
    ψ      = (α, β, T21, T22, A1, A2, T1, TE)
    X      = _biexp_epg_model_f64!(epg.dc, epg, ψ)
    Xmax   = zero(eltype(X))
    @simd for i in eachindex(X)
        Xmax = max(X[i], Xmax)
    end
    Xscale = s / Xmax
    @simd for i in eachindex(X)
        X[i] *= Xscale
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
    logs  ~ Distributions.Normal(0.0, 0.5)
    X     = biexp_epg_model_scaled!(work, α, β, η, δ1, δ2, logϵ, logs)
    logϵs = logϵ + logs
    ϵs    = exp(logϵs)
    for i in 1:length(Y)
        logℒ = -neglogL_rician(Y[i], X[i], logϵs)
        Turing.@addlogprob! logℒ
        # Y[i] ~ Rician(X[i], ϵs)
    end
end

function mcmc_biexp_epg(
        phys;
        img_idx         = 1,
        dataset         = :val,
        save            = true,
        checkpoint      = save,
        total_chains    = Colon(),
        checkpoint_freq = 2048, # checkpoint every `checkpoint_freq` iterations
        progress_freq   = 15.0, # update progress bar every `progress_freq` seconds
    )

    file_prefix(ischeckpoint) = (ischeckpoint ? "checkpoint_" : "") * "image-$(img_idx)_dataset-$(dataset)"
    file_suffix(chains) = "$(lpad(chains[1].col, 7, '0'))-to-$(lpad(chains[end].col, 7, '0'))"

    function save_chains(chains; ischeckpoint)
        filename = file_prefix(ischeckpoint) * "_" * file_suffix(chains) * ".jld2"
        FileIO.save(filename, Dict("chains" => chains))
    end

    function save_dataframe(chains; ischeckpoint)
        filename = file_prefix(ischeckpoint) * "_" * file_suffix(chains) * ".csv"
        chains_df = mapreduce(vcat, chains) do (col, index, chain)
            df = DataFrame(chain)
            df.dataset_col = fill(col, DataFrames.nrow(df))
            df.image_x = fill(index[1], DataFrames.nrow(df))
            df.image_y = fill(index[2], DataFrames.nrow(df))
            df.image_z = fill(index[3], DataFrames.nrow(df))
            return df
        end
        CSV.write(filename, chains_df)
    end

    img              = phys.images[img_idx]
    ydata            = img.partitions[dataset] .|> Float64
    yindices         = img.indices[dataset]
    work_buffers     = [mcmc_biexp_epg_work_factory(phys, img) for _ in 1:Threads.nthreads()]

    total_chains     = total_chains == Colon() ? size(ydata, 2) : total_chains
    chains           = Any[nothing for _ in 1:total_chains]
    checkpoint_cb    = !(save && checkpoint) ? nothing : Js -> save_dataframe(view(chains, Js); ischeckpoint = true)

    Turing.setprogress!(false)
    Turing.setchunksize(7)

    foreach_with_progress(eachindex(chains); checkpoint_cb, checkpoint_freq, dt = progress_freq) do J
        work  = work_buffers[Threads.threadid()]
        Y     = ydata[:,J]
        model = mcmc_biexp_epg_model!(work, Y)
        chain = DECAES.tee_capture(suppress_terminal = true, suppress_logfile = true) do io
            Turing.sample(model, Turing.NUTS(0.65), 100)
        end
        chains[J] = (; col = J, index = yindices[J], chain = chain)
    end

    save && save_dataframe(chains; ischeckpoint = false)
    save && save_chains(chains; ischeckpoint = false)

    return chains
end

function _test_mcmc()
    # img = phys.images[img_idx]
    # Y = img.partitions[dataset] .|> Float64
    # X = img.meta[:pseudolabels][dataset][:signalfit] .|> Float64
    # θ = img.meta[:pseudolabels][dataset][:theta] .|> Float64
    # Z = img.meta[:pseudolabels][dataset][:latent] .|> Float64

    # J, _ = sample_maybeshuffle(1:size(Y,2); shuffle = true, samples, seed)
    # J = J[1] #TODO
    # Y = Y[:,J] .|> Float64
    # X = X[:,J] .|> Float64
    # θ = θ[:,J] .|> Float64
    # logϵ = Z[:,J] .|> Float64 # Z = log(ϵ)
    # logs = log.(inv.(maximum(mean_rician.(X, exp.(logϵ)); dims = 1))) .|> Float64
    # init_params = [θmarginalized(phys, θ); logϵ; logs]
    # plot([Y biexp_epg_model_scaled!(init_params...)]) |> display

    # df = DataFrame(chains[1][2])
    # α, β, η, δ1, δ2, logϵ, logs = df[end, [:α, :β, :η, :δ1, :δ2, :logϵ, :logs]]
    # xfit = biexp_epg_model_scaled!(work, α, β, η, δ1, δ2, logϵ, logs)
    # plot([ydata[:,1] xfit]) |> display
    # plot(chains[1][2]) |> display

    return chains
end

####
#### MLE inference
####

function _test_mle_biexp_epg_noise_only(;
        initial_ϵ = true, img_idx = 1, samples = 1024, seed = 0
    )
    img = Main.phys.images[img_idx]
    dataset = :val
    Y = img.partitions[dataset]
    X = img.meta[:pseudolabels][dataset][:signalfit]
    θ = img.meta[:pseudolabels][dataset][:theta]
    Z = img.meta[:pseudolabels][dataset][:latent]
    J, _ = sample_maybeshuffle(1:size(Y,2); shuffle = true, samples, seed)
    Y = Y[:,J] |> to32
    X = X[:,J] |> to32
    θ = θ[:,J] |> to32
    ϵ = exp.(Z[:,J] |> to32) # Z = log(ϵ)

    @info "Neither frozen"
    initial_guess, results = mle_biexp_epg_noise_only(X, Y, initial_ϵ ? ϵ : nothing; freeze_ϵ = false, freeze_s = false)

    @info "scale frozen"
    initial_guess, results = mle_biexp_epg_noise_only(X, Y, initial_ϵ ? ϵ : nothing; freeze_ϵ = false, freeze_s = true)
    @assert vec(log.(initial_guess.s)) ≈ vec(results.logscale)

    @info "epsilon frozen"
    initial_guess, results = mle_biexp_epg_noise_only(X, Y, initial_ϵ ? ϵ : nothing; freeze_ϵ = true, freeze_s = false)
    @assert vec(log.(initial_guess.ϵ)) ≈ vec(results.logepsilon)

    @info "NegLogLikelihood"
    ℓ = NegLogLikelihood(Main.phys, Main.models["genatr"], Y, X, ϵ)
    sum(ℓ) / size(ℓ,2) |> mean |> display
end

function mle_biexp_epg_noise_only(
        X::AbstractVecOrMat,
        Y::AbstractVecOrMat,
        initial_ϵ     = nothing,
        initial_s     = nothing;
        freeze_ϵ      = false,
        freeze_s      = false,
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
    @assert !(freeze_ϵ && freeze_s) # must be optimizing at least one

    dryrun && let
        I, _ = sample_maybeshuffle(1:size(Y,2); shuffle = dryrunshuffle, samples = dryrunsamples, seed = dryrunseed)
        X, Y = X[:,I], Y[:,I]
        (initial_ϵ !== nothing) && (initial_ϵ = initial_ϵ[:,I])
        (initial_s !== nothing) && (initial_s = initial_s[:,I])
    end

    if initial_ϵ === nothing
        initial_ϵ = sqrt.(mean(abs2, X .- Y; dims = 1))
    end
    if initial_s === nothing
        initial_s = inv.(maximum(mean_rician.(X, initial_ϵ); dims = 1))
        # initial_s = ones_similar(X, 1, size(X,2))
    end
    initial_loss  = sum(neglogL_rician.(Y, initial_s .* X, log.(initial_s .* initial_ϵ)); dims = 1)
    initial_guess = (
        ϵ = initial_ϵ |> arr64,
        s = initial_s |> arr64,
        ℓ = initial_loss |> arr64,
    )
    X, Y = (X, Y) .|> arr64

    # Number of non-frozen variables to optimize
    noptvars = !freeze_ϵ + !freeze_s

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
            logϵ  = freeze_ϵ ? work.logϵ0[] : freeze_s ? x[1] : x[1]
            logs  = freeze_s ? work.logs0[] : freeze_ϵ ? x[1] : x[2]
            s     = exp(logs)
            logϵi = logs + logϵ # log(s*ϵ)
            ℓ     = zero(D)
            @simd for i in eachindex(work.Xj) #TODO @avx?
                yi = work.Yj[i]
                νi = s * work.Xj[i]
                ℓ += neglogL_rician(yi, νi, logϵi) # Rician negative log likelihood
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

    #= Benchmarking
    work_spaces[1].Xj .= X[:,1]
    work_spaces[1].Yj .= Y[:,1]
    work_spaces[1].x0 .= freeze_ϵ ? log.(initial_guess.s[1:1,1]) : freeze_s ? log.(initial_guess.ϵ[1:1,1]) : log.(vcat(initial_guess.ϵ[1:1,1], initial_guess.s[1:1,1]))
    @info "Calling function..."; l = f( work_spaces[1], work_spaces[1].x0 ); @show l
    @info "Calling gradient..."; g = zeros(noptvars); l = fg!( work_spaces[1], work_spaces[1].x0, g ); @show l, g
    @info "Timing function..."; @btime( $f( $( work_spaces[1] ), $( work_spaces[1].x0 ) ) )
    @info "Timing gradient..."; @btime( $fg!( $( work_spaces[1] ), $( work_spaces[1].x0 ), $( g ) ) )
    =#

    start_time = time()
    batches = batch_size === Colon() ? [1:size(Y,2)] : Iterators.partition(1:size(Y,2), batch_size)
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Xj               .= X[:,j]
                work.Yj               .= Y[:,j]
                work.logϵ0[]           = log(initial_guess.ϵ[1,j])
                work.logs0[]           = log(initial_guess.s[1,j])
                work.x0               .= freeze_ϵ ? work.logs0[] : freeze_s ? work.logϵ0[] : (work.logϵ0[], work.logs0[])
                work.opt.min_objective = (x, g) -> fg!(work, x, g)

                solvetime              = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                results.logepsilon[j]  = freeze_ϵ ? work.logϵ0[] : freeze_s ? minx[1] : minx[1]
                results.logscale[j]    = freeze_s ? work.logs0[] : freeze_ϵ ? minx[1] : minx[2]
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
        α, β, η, δ1, δ2 = θ[1], θ[2], θ[3], θ[4], θ[5]
        t   = 1e-12
        δ1′ = clamp(logτ2lo + (logτ2hi - logτ2lo) * δ1, logτ2lo′ + t, logτ2hi′ - t)
        δ2′ = clamp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)), logτ2lo′ + t, logτ2hi′ - t)
        z1  = (δ1′ - logτ2lo′) / (logτ2hi′ - logτ2lo′)
        z2  = ((δ2′ - logτ2lo′) / (logτ2hi′ - logτ2lo′) - z1) / (1 - z1)
        return α, β, η, z1, z2
    end
end

function mle_biexp_epg_x_to_θ(work::W, x::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack logτ2lo, logτ2hi, logτ2lo′, logτ2hi′, δ0 = work
        α, β, η, z1, z2 = x[1], x[2], x[3], x[4], x[5]
        δ1 = ((logτ2lo′ - logτ2lo) + (logτ2hi′ - logτ2lo′) * z1) / (logτ2hi - logτ2lo)
        δ2 = (((logτ2lo′ - logτ2lo) + (logτ2hi′ - logτ2lo′) * (z1 + z2 * (1 - z1))) / (logτ2hi - logτ2lo) - δ1) / (1 - δ1)
        return α, β, η, δ1, δ2, δ0
    end
end

function mle_biexp_epg_θ_to_ψ(work::W, θ::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack logτ2lo, logτ2hi, logτ1lo, logτ1hi = work
        α, β, η, δ1, δ2, δ0 = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6]
        T21 = exp(logτ2lo + (logτ2hi - logτ2lo) * δ1)
        T22 = exp(logτ2lo + (logτ2hi - logτ2lo) * (δ1 + δ2 * (1 - δ1)))
        T1 = exp(logτ1lo + (logτ1hi - logτ1lo) * δ0)
        return α, β, T21, T22, η, 1-η, T1, one(T1)
    end
end

function mle_biexp_epg_x_regularization(work::W, x::A) where {W, A <: VecOrTupleMaybeDualF64}
    @inbounds begin
        @unpack xlo, xhi, σreg = work
        α,   β,   η,   z1,   z2   = x[1],   x[2],   x[3],   x[4],   x[5]   # flip/refcon angles, mwf, and relative log T2 short/long
        αlo, βlo, ηlo, z1lo, z2lo = xlo[1], xlo[2], xlo[3], xlo[4], xlo[5] # parameter lower bounds
        αhi, βhi, ηhi, z1hi, z2hi = xhi[1], xhi[2], xhi[3], xhi[4], xhi[5] # parameter upper bounds
        # Regularization is negative log-likelihood of a sum of Gaussian distributions centred on parameter interval lower/upper bounds.
        # This biases parameters to prefer to be at one endpoint or the other in cases where the solution is ill-posed.
        # Gaussian widths are equal to σreg * (interval width). Normalization constant is discarded.
        R = (
            ((α - αhi) / (αhi - αlo))^2 + # flip angle biased toward 180 deg
            ((β - βhi) / (βhi - βlo))^2 + # refcon angle biased toward 180 deg
            ((η - ηlo) / (ηhi - ηlo))^2 + # mwf biased toward 0%
            ((z1 - z1lo) / (z1hi - z1lo))^2 + # (relative-)log T2 short biased lower
            ((z2 - z2hi) / (z2hi - z2lo))^2   # (relative-)log T2 long biased higher
        ) / (2 * σreg^2) # regularization strength
        return R
    end
end

@inline mle_biexp_epg_work(work::W, ::Type{Float64}) where {W} = work.epg
@inline mle_biexp_epg_work(work::W, ::Type{D}) where {W, D <: Dual64} = work.∇epg

function f_mle_biexp_epg!(work::W, x::Vector{D}) where {W, D <: MaybeDualF64}
    @inbounds begin
        θ      = mle_biexp_epg_x_to_θ(work, x) # θ = α, β, η, δ1, δ2, δ0
        logϵ   = x[end-1]
        logs   = x[end]
        s      = exp(logs)
        ψ      = mle_biexp_epg_θ_to_ψ(work, θ) # ψ = alpha, refcon, T2short, T2long, Ashort, Along, T1, TE
        epg    = mle_biexp_epg_work(work, D)
        X      = _biexp_epg_model_f64!(epg.dc, epg, ψ)
        Xmax   = zero(D)
        @simd for i in eachindex(X) #TODO @avx?
            Xmax = max(X[i], Xmax)
        end
        Xscale = s / Xmax # normalize X to maximum 1 and scale by s; hoist outside loop
        logϵi  = logs + logϵ # hoist outside loop
        ℓ      = zero(D)
        @simd for i in eachindex(work.Y) #TODO @avx?
            νi = Xscale * X[i]
            ℓ += neglogL_rician(work.Y[i], νi, logϵi) # Rician negative log likelihood
        end
        R      = mle_biexp_epg_x_regularization(work, x)
        work.neglogL[] = ForwardDiff.value(ℓ)
        work.reg[] = ForwardDiff.value(R)
        return ℓ + R
    end
end

function fg_mle_biexp_epg!(work::W, x::Vector{Float64}, g::Vector{Float64}) where {W}
    if length(g) > 0
        # simple_fd_gradient!(g, _x -> f_mle_biexp_epg!(work, _x), x, work.xlo, work.xhi)
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
        rice::RicianCorrector,
        cvae::CVAE,
        img::CPMGImage;
        data_source        = :image, # One of :image, :simulated
        data_subset        = :mask,  # One of :mask, :val, :train, :test
        noisebounds        = (-Inf, 0.0),
        scalebounds        = (-Inf, Inf),
        batch_size         = Colon(),
        maxiter            = 10,
        mode               = :mode,  # :maxlikelihood, :mean
        verbose            = true,
        dryrun             = false,
        dryrunsamples      = batch_size,
        dryrunshuffle      = true,
        dryrunseed         = 0,
    )
    @assert data_source ∈ (:image, :simulated)
    @assert data_subset ∈ (:mask, :val, :train, :test)

    # MLE for whole image of simulated data
    image_indices = img.indices[data_subset]
    if dryrun
        image_indices, _ = sample_maybeshuffle(image_indices; shuffle = dryrunshuffle, samples = dryrunsamples, seed = dryrunseed)
    end
    image_data = img.data[image_indices, :] |> to32 |> permutedims

    if data_source === :image
        Y = arr64(image_data)
        Ymeta = MetaCPMGSignal(phys, img, image_data)
    else # data_source === :simulated
        mock_image_state = batched_posterior_state(MetaCPMGSignal(phys, img, image_data))
        mock_image_data  = sampleX̂(rice, mock_image_state.X, mock_image_state.Z)
        Y = arr64(mock_image_data)
        Ymeta = MetaCPMGSignal(phys, img, mock_image_data)
    end

    batches = batch_size === Colon() ? [1:size(signal(Ymeta),2)] : Iterators.partition(1:size(signal(Ymeta),2), batch_size)
    initial_guess = mapreduce((x,y) -> map(hcat, x, y), batches) do batch
        posterior_state(
            phys, rice, cvae, Ymeta[:,batch];
            alpha = 0.0, miniter = 1, maxiter, verbose, mode,
        )
    end
    initial_guess = setindex!!(initial_guess, exp.(clamp.(mean(log.(initial_guess.ϵ); dims = 1), noisebounds...)), :ϵ)
    initial_guess = setindex!!(initial_guess, clamp.(inv.(maximum(mean_rician.(initial_guess.X, initial_guess.ϵ); dims = 1)), scalebounds...), :s)
    initial_guess = map(arr64, initial_guess)
    return initial_guess
end

function mle_biexp_epg(
        phys::BiexpEPGModel,
        rice::RicianCorrector,
        cvae::CVAE,
        img::CPMGImage;
        batch_size         = 128 * Threads.nthreads(),
        initial_guess_args = Dict{Symbol,Any}(),
        initial_guess_only = false,
        noisebounds        = (-Inf, 0.0),
        scalebounds        = (-Inf, Inf),
        sigma_reg          = 0.5,
        verbose            = true,
        opt_alg            = :LD_SLSQP, # Rough algorithm ranking: [:LD_SLSQP, :LD_LBFGS, :LD_CCSAQ, :LD_AUGLAG, :LD_MMA] (Note: :LD_LBFGS fails to converge with tolerance looser than ~ 1e-4)
        opt_args           = Dict{Symbol,Any}(),
    )

    initial_guess = mle_biexp_epg_initial_guess(phys, rice, cvae, img; noisebounds, scalebounds, verbose = false, initial_guess_args...)

    # Optionally return initial guess only
    if initial_guess_only
        return initial_guess, nothing
    end

    num_optvars   = nmarginalized(phys) + 2
    lower_bounds  = [θmarginalized(phys, θlower(phys)); noisebounds[1]; scalebounds[1]] |> arr64
    upper_bounds  = [θmarginalized(phys, θupper(phys)); noisebounds[2]; scalebounds[2]] |> arr64

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
            δ0       = initial_guess.θ[end,1] |> Float64,
            epg      = BiexpEPGModelWork(phys, Float64),
            ∇epg     = BiexpEPGModelWork(phys, ForwardDiff.Dual{Nothing, Float64, num_optvars}),
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
        logepsilon = fill(NaN, num_problems),
        logscale   = fill(NaN, num_problems),
        loss       = fill(NaN, num_problems),
        reg        = fill(NaN, num_problems),
        retcode    = fill(NaN, num_problems),
        numevals   = fill(NaN, num_problems),
        solvetime  = fill(NaN, num_problems),
    )

    #= Benchmarking
    work              = work_spaces[1]
    work.Y           .= Y[:,1]
    work.x0[1:end-2] .= mle_biexp_epg_θ_to_x(work, view(initial_guess.θ, :, 1))
    work.x0[end-1]    = log(initial_guess.ϵ[1])
    work.x0[end]      = log(initial_guess.s[1])
    g                 = zeros(num_optvars)
    # @show ForwardDiff.gradient!(work.∇res, x -> f_mle_biexp_epg!(work, x), work.x0, work.∇cfg)
    # @show fg_mle_biexp_epg!(work, work.x0, g); @show(g)
    @info "Timing function..."; @btime( f_mle_biexp_epg!($work, $(work.x0)) ) |> display
    @info "Timing ForwardDiff gradient..."; @btime( ForwardDiff.gradient!($(work.∇res), x -> f_mle_biexp_epg!($work, x), $(work.x0), $(work.∇cfg)) ) |> display
    @info "Timing gradient..."; @btime( fg_mle_biexp_epg!($work, $(work.x0), $g) ) |> display
    return nothing
    =#

    start_time = time()
    batches = Iterators.partition(1:num_problems, batch_size)
    BLAS.set_num_threads(1) # Prevent BLAS from stealing julia threads

    for (batchnum, batch) in enumerate(batches)
        batchtime = @elapsed Threads.@sync for j in batch
            Threads.@spawn @inbounds begin
                work = work_spaces[Threads.threadid()]
                work.Y                 .= initial_guess.Y[:,j]
                work.x0[1:end-2]       .= mle_biexp_epg_θ_to_x(work, view(initial_guess.θ, :, j))
                work.x0[end-1]          = log(initial_guess.ϵ[j])
                work.x0[end]            = log(initial_guess.s[j])
                work.opt.min_objective  = (x, g) -> fg_mle_biexp_epg!(work, x, g)
                initial_guess.ℓ[j]      = f_mle_biexp_epg!(work, work.x0) #TODO: can save a small amount of time by deleting this, but probably worth it, since previous ℓ[j] is an approximation from posterior_state

                solvetime               = @elapsed (minf, minx, ret) = NLopt.optimize(work.opt, work.x0)
                f_mle_biexp_epg!(work, minx) # update `work` with solution

                results.signalfit[:,j] .= work.epg.dc ./ maximum(work.epg.dc)
                results.theta[:,j]     .= mle_biexp_epg_x_to_θ(work, minx)
                results.logepsilon[j]   = minx[end-1]
                results.logscale[j]     = minx[end]
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
            " -- initial loss: $(round(mean(initial_guess.ℓ[1,1:batch[end]]); digits = 2))" *
            " -- loss: $(round(mean(results.loss[1:batch[end]]); digits = 2))" *
            " -- reg: $(round(mean(results.reg[1:batch[end]]); digits = 2))"
    end

    BLAS.set_num_threads(Threads.nthreads()) # Reset BLAS threads

    return initial_guess, results
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
        mle_image_results = readdir(Glob.glob"mle-image-mask-results-final-*.mat", mle_image_path) |> only |> DECAES.MAT.matread
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
        mle_sim_data = readdir(Glob.glob"mle-simulated-mask-data-*.mat", mle_sim_path) |> only |> DECAES.MAT.matread
        mle_sim_results = readdir(Glob.glob"mle-simulated-mask-results-final-*.mat", mle_sim_path) |> only |> DECAES.MAT.matread

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
