# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
include(joinpath(@__DIR__, "src", "mmd_math.jl"))
include(joinpath(@__DIR__, "src", "mmd_utils.jl"))

####
#### Global MMD minimization
####

function beta_bounds(T = Float64;
        ntheta = settings["data"]["ntheta"]::Int,
        nterms = settings["prior"]["nterms"]::Int,
    )
    function mvnormal_term_bounds()
        #mean_lb= T[150.0,  50.0, 0.001,    5.0,  100.0, 0.001,    5.0,  100.0]
        #mean_lb= T[150.0, 150.0, 0.001,    5.0,  100.0, 0.001,    5.0,  100.0]
        #mean_ub= T[180.0, 180.0,   1.0, 1000.0, 3000.0,   1.0, 1000.0, 3000.0]
        #mean_lb= T[ 50.0,  50.0,  1e-4,    5.0,  999.0,  1e-4,    5.0,  999.0]
        #mean_ub= T[180.0, 180.0,   1.0,  500.0, 1000.0,   1.0, 2000.0, 1000.0]
        mean_lb = T[120.0, 120.0,    8.0,    8.0, 0.0]
        mean_ub = T[180.0, 180.0, 1000.0, 1000.0, 1.0]
        means = collect(zip(mean_lb, mean_ub))
        @assert ntheta == length(means)

        cov_bd = T(1/ntheta)
        cov_fact_lb = fill(-cov_bd, ntheta^2)
        cov_fact_ub = fill(+cov_bd, ntheta^2)
        cov_facts = collect(zip(cov_fact_lb, cov_fact_ub))
        weights = [(T(0.0), T(1.0))] # prior weights
        vcat(means, cov_facts, weights)
    end
    beta_bounds = [
        reduce(vcat, [mvnormal_term_bounds() for _ in 1:nterms]);
        [(T(0.0), T(0.02))]; # noise bounds
    ]
    return beta_bounds
end
beta_sampler(args...; kwargs...) = broadcast(bound -> bound[1] + (bound[2]-bound[1]) * rand(typeof(bound[1])), beta_bounds(args...; kwargs...))

function param_sampler(β::AbstractVector{T};
        ntheta = settings["data"]["ntheta"]::Int,
        nterms = settings["prior"]["nterms"]::Int,
    ) where {T}
    ndistparams = 1 + ntheta + ntheta^2
    βterms = @views(β[1:end-1])
    βterms = reshape(βterms, ndistparams, nterms)
    @assert length(βterms) ÷ ndistparams == nterms
    @assert length(βterms) == nterms * ndistparams
    means_idx = 1:ntheta
    cov_fact_idx = means_idx[end] .+ (1:ntheta^2)
    θ_bounds = beta_bounds(T)[1:ntheta]
    θ_mean = mean.(θ_bounds)
    θ_scale = (x -> (x[2]-x[1])/2).(θ_bounds)
    scale_back = θ -> (θ .- θ_mean) ./ θ_scale
    scale_forw = θ -> θ_scale .* θ .+ θ_mean
    function make_mvnormal(j)
        μ = scale_back(βterms[means_idx, j])
        A = reshape(βterms[cov_fact_idx, j], ntheta, ntheta)
        Σ = A' * A
        return MvNormal(μ, Σ)
    end
    distbns = make_mvnormal.(1:nterms)
    prior_weights = βterms[end,:] |> x -> x./sum(x)

    mvnormal_mixture = MixtureModel(distbns, prior_weights)
    function sampler()
        η = rand(mvnormal_mixture)
        θ = scale_forw(tanh.(η))
        # Enforce parameter bounds
        for i in eachindex(θ)
            θ[i] = clamp(θ[i], θ_bounds[i]...) # in case of round-off error
            # @assert θ_bounds[i][1] <= θ[i] <= θ_bounds[i][2] # `tanh` should handle this
        end
        # if θ[3] > θ[4]
        #     # Sort short/long components
        #     θ[3], θ[4], θ[5] = θ[4], θ[3], 1-θ[5]
        # end
        ϵ = β[end:end] # noise variables
        return θ, ϵ
    end
    return sampler
end

function mmd_signal_model(β::AbstractVector{T}, Y::AbstractMatrix{T};
        logsigma = T(settings["prior"]["logsigma"]),
    ) where {T}
    sampler = param_sampler(β)
    signal_work = signal_model_work(T)
    X = similar(Y)
    for Xj in eachcol(X)
        θ, ϵ = sampler()
        Xj .= signal_model!(signal_work, θ, ϵ) # compute signal
    end
    γ = inv(T(2*exp(2*logsigma)))
    # mse = (x,y) -> mean(abs2, x .- y)
    # kernel = (x,y) -> exp(-γ*mse(x,y))
    kernel = Δ -> exp(-γ*Δ)
    MMDsq = mmd(kernel, X, Y)
    return MMDsq
end

function plotting_callback(
        β::AbstractVector{T},
        image::Array{T,4};
        out = settings["data"]["out"]::String,
        ntheta = settings["data"]["ntheta"]::Int,
        theta_labels = settings["data"]["theta_labels"]::Vector{String},
        batchsize = settings["prior"]["batchsize"]::Int,
        nbatches = settings["prior"]["nbatches"]::Int,
        logsigma = T(settings["prior"]["logsigma"]),
    ) where {T}
    sampler_fun = param_sampler(β)
    thetas = reduce(hcat, [sampler_fun()[1] for _ in 1:100_000])
    example_data = signal_data(image, batchsize)
    example_model = reduce(hcat, [signal_model(sampler_fun()...) for _ in 1:batchsize])

    phist = plot(map((lab,row,xl) -> histogram(row; nbins = 50, yticks = :none, lab = lab, xlims = xl), theta_labels, eachrow(thetas), beta_bounds()[1:ntheta])...; xrot = 45); display(phist)
    psamp = plot!(plot(example_model; c = :red, lw = 0.5, leg = :none), example_data; c = :blue, lw = 0.5, title = "model (red) vs. data (blue)"); display(psamp)
    pheat = mmd_heatmap(example_model, example_data, exp(logsigma)); display(pheat)
    map(ext -> savefig(phist, joinpath(out, "thetas_hist.$ext")), ["pdf", "png"])
    map(ext -> savefig(psamp, joinpath(out, "signal_sample.$ext")), ["pdf", "png"])
    map(ext -> savefig(pheat, joinpath(out, "mmd_heatmap.$ext")), ["pdf", "png"])
    return nothing
end

function make_loss(image::Array{T,4};
        batchsize = settings["prior"]["batchsize"]::Int,
        nbatches = settings["prior"]["nbatches"]::Int,
    ) where {T}
    Y = signal_data(image)
    Yidx = sample(1:size(Y,2), batchsize * nbatches; replace = false)
    Ysets = [Y[:,J] for J in Iterators.partition(Yidx, batchsize)]
    @assert length(Ysets) == nbatches

    best_loss = Ref(T(Inf))
    fun_count = Ref(0)
    MMDsq = zeros(T, nbatches)
    function(β)
        fun_count[] += 1
        MMDsq .= 0
        BLAS.set_num_threads(1)
        Threads.@threads for i in 1:nbatches
            MMDsq[i] = mmd_signal_model(β, Ysets[i])
        end
        BLAS.set_num_threads(Threads.nthreads())

        loss = mean(MMDsq)
        # loss = median(MMDsq)
        if loss < best_loss[]
            best_loss[] = loss
            if fun_count[] >= 100
                plotting_callback(β, image)
            end
        end
        return loss
    end
end

function test_beta_mmd_bbopt(image::Array{T,4};
        method = settings["prior"]["method"]::String,
        maxtime = T(settings["prior"]["maxtime"]),
    ) where {T}
    res = bboptimize(
        make_loss(image);
        SearchRange = beta_bounds(T),
        MaxFuncEvals = typemax(Int),
        TraceMode = :verbose,
        Method = method,
        MaxTime = maxtime,
    )
    return res
end

function test_beta_mmd_optim(image::Array{T,4}, β₀::AbstractVector{T}) where {T}
    bounds = beta_bounds(T)
    lower = (x->x[1]).(bounds)
    upper = (x->x[2]).(bounds)
    ℓ = make_loss(image)
    count = 0
    res = optimize(lower, upper, β₀, Fminbox(NelderMead())) do β
        rng = Random.seed!(0)
        loss = ℓ(β)
        Random.seed!(rng)
        count += 1
        return loss
    end
    return res
end

function test_beta_mmd_flux(image::Array{T,4}, β₀::AbstractVector{T}) where {T}
    bounds = beta_bounds(T)
    lower = (x->x[1]).(bounds)
    upper = (x->x[2]).(bounds)
    ℓ = make_loss(image)

    gradient = similar(β₀)
    function gradloss!(dβ, β)
        rng = Random.seed!(0)
        loss = ℓ(β)
        for i in eachindex(dβ, β)
            βi = β[i]
            delta = sqrt(eps(T)) * (upper[i] - lower[i])
            if βi + delta <= upper[i]
                β[i] = βi + delta
                Random.seed!(0)
                loss_plus = ℓ(β)
                dβ[i] = (loss_plus - loss) / delta
            else
                β[i] = βi - delta
                Random.seed!(0)
                loss_minus = ℓ(β)
                dβ[i] = (loss - loss_minus) / delta
            end
            β[i] = βi
        end
        Random.seed!(rng)
        return loss
    end

    loss = gradloss!(gradient, β₀)

    return nothing
end

#=
let
    res = test_beta_mmd_bbopt(image)
    DECAES.MAT.matwrite(joinpath(settings["data"]["out"], "bboptim_results.mat"), make_save_dict(res))
    β = best_candidate(res)
    plotting_callback(β, image)
end
=#

nothing
