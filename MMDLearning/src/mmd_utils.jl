####
#### Settings file loading
####

function load_settings(
        default_settings_file = joinpath(@__DIR__, "default_settings.toml"),
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

####
#### Signal model
####

function signal_data(
        image::Array{T,4},
        batchsize = nothing;
        threshold# = T(opts.Threshold)
    ) where {T}
    first_echo = filter!(>(threshold), image[:,:,:,1][:])
    q1 = quantile(first_echo, 0.30)
    q2 = quantile(first_echo, 0.99)
    Is = findall(I -> q1 <= image[I,1] <= q2, CartesianIndices(size(image)[1:3]))
    Y  = image[Is, :]' |> copy
    # map(_ -> display(plot(image[rand(Is),:])), 1:5)
    # histogram(first_echo) |> p -> vline!(p, [q1, q2]) |> display
    Y ./= sum(Y; dims=1)
    return batchsize === nothing ? Y : sample_columns(Y, batchsize)
end

function theta_bounds(T = Float64; ntheta::Int)
    if ntheta == 4
        theta_labels = ["alpha", "T2short", "dT2", "Ashort"]
        # ["refcon", "alpha", "T2short", "dT2", "Ashort"]
        theta_lb = T[ 50.0,    8.0,    8.0, 0.0]
        theta_ub = T[180.0, 1000.0, 1000.0, 1.0]
        theta_bd = collect(zip(theta_lb, theta_ub))
    elseif ntheta == 5
        # ["refcon", "alpha", "T2short", "dT2", "Ashort"]
        theta_lb = T[120.0, 120.0,    8.0,    8.0, 0.0]
        theta_ub = T[180.0, 180.0, 1000.0, 1000.0, 1.0]
        theta_bd = collect(zip(theta_lb, theta_ub))
    else
        error("Number of labels must be 4 or 5")
    end
    @assert ntheta == length(theta_bd)
    return theta_bd
end
theta_sampler(args...; kwargs...) = broadcast(bound -> bound[1] + (bound[2]-bound[1]) * rand(typeof(bound[1])), theta_bounds(args...; kwargs...))

function signal_theta_error(theta, thetahat)
    dtheta = (x -> x[2] - x[1]).(theta_bounds(eltype(theta); ntheta = size(theta,1)))
    return abs.((theta .- thetahat)) ./ dtheta
end

noise_model!(buffer::AbstractVector{T}, signal::AbstractVector{T}, ϵ::AbstractVector{T}) where {T} = (randn!(buffer); buffer .*= ϵ .* signal[1]; buffer)
noise_model(signal::AbstractVector, ϵ::AbstractVector) = ϵ .* signal[1] .* randn(eltype(signal), length(signal))

function signal_model_work(T = Float64; nTE::Int)
    epg_work = DECAES.EPGdecaycurve_work(T, nTE)
    signal = zeros(T, nTE)
    buf = zeros(T, nTE)
    real_noise = zeros(T, nTE)
    imag_noise = zeros(T, nTE)
    return @ntuple(epg_work, signal, buf, real_noise, imag_noise)
end

function signal_model!(
        work,
        θ::AbstractVector{T},
        ϵ::Union{AbstractVector, Nothing} = nothing;
        TE
    ) where {T}
    @unpack epg_work, signal, real_noise, imag_noise = work
    # refcon, alpha, T2short, dT2, Ashort = θ[1], θ[2], θ[3]/1000, θ[4]/1000, θ[5]
    alpha, T2short, dT2, Ashort = θ[1], θ[2]/1000, θ[3]/1000, θ[4]
    refcon  = T(180.0)
    T2long  = T2short + dT2
    Along   = 1-Ashort
    T1short = T(1000.0)/1000
    T1long  = T(1000.0)/1000

    signal  .= Ashort .* DECAES.EPGdecaycurve!(epg_work, alpha, T(TE), T2short, T1short, refcon) # short component
    signal .+= Along  .* DECAES.EPGdecaycurve!(epg_work, alpha, T(TE), T2long,  T1long,  refcon) # long component
    signal ./= sum(signal) # normalize

    # Add noise to "real" and "imag" channels in quadrature
    if !isnothing(ϵ)
        if eltype(ϵ) === eltype(θ)
            noise_model!(real_noise, signal, ϵ) # populate real_noise
            noise_model!(imag_noise, signal, ϵ) # populate imag_noise
            signal .= sqrt.((signal .+ real_noise).^2 .+ imag_noise.^2)
            signal ./= sum(signal)
        else
            # Add forward-differentiable noise to "real" and "imag" channels in quadrature
            signal = sqrt.((signal .+ noise_model(signal, ϵ)).^2 .+ noise_model(signal, ϵ).^2)
            signal ./= sum(signal)
        end
    end

    return signal
end

function signal_model!(
        work,
        θ::AbstractMatrix{T},
        ϵ::Union{AbstractVector, Nothing} = nothing;
        kwargs...
    ) where {T}
    @unpack signal = work
    X = zeros(length(signal), size(θ,2))
    @uviews θ X for j in 1:size(θ,2)
        signal_model!(work, θ[:,j], ϵ; kwargs...)
        X[:,j] .= signal
    end
    return X
end

signal_model(θ::AbstractVecOrMat{T}, ϵ::Union{AbstractVector, Nothing} = nothing; nTE::Int, TE) where {T} = signal_model!(signal_model_work(T; nTE = nTE), θ, ϵ; TE = T(TE))

function mutate_signal(Y::AbstractVecOrMat; meanmutations::Int = 0)
    if meanmutations <= 0
        return Y
    end
    nrow = size(Y, 1)
    p = meanmutations / nrow
    return Y .* (rand(size(Y)...) .> p)
end

####
#### MMD plotting
####

function mmd_heatmap(X, Y, σ; skipdiag = true)
    γ = inv(2*σ^2)
    k = Δ -> exp(-γ*Δ)

    # compute mmd
    work = mmd_work(X, Y)
    mmd = mmd!(work, k, X, Y)

    # recompute with diag for plotting
    @unpack Kxx, Kyy, Kxy = work
    if !skipdiag
        kernel_pairwise!(Kxx, k, X, X, Val(false))
        kernel_pairwise!(Kyy, k, Y, Y, Val(false))
        kernel_pairwise!(Kxy, k, X, Y, Val(false))
    end

    s = x -> string(round(x; sigdigits = 3))
    m = size(Kxx, 1)
    K = [Kxx Kxy; Kxy' Kyy]
    Kplot = (x -> x == 0 ? 0.0 : log10(x)).(K[end:-1:1,:])
    p = heatmap(Kplot; title = "m*MMD = $(s(m*mmd)), sigma = $(s(σ))", clims = (max(minimum(Kplot), -10), 0))

    return p
end

function mmd_witness(X, Y, σ; skipdiag = false)
    γ = inv(2*σ^2)
    k = Δ -> exp(-γ*Δ)

    # compute mmd
    work = mmd_work(X, Y)
    mmd = mmd!(work, k, X, Y)

    # recompute with diag for plotting
    @unpack Kxx, Kyy, Kxy = work
    if !skipdiag
        kernel_pairwise!(Kxx, k, X, X, Val(false))
        kernel_pairwise!(Kyy, k, Y, Y, Val(false))
        kernel_pairwise!(Kxy, k, X, Y, Val(false))
    end

    s = x -> string(round(x; sigdigits = 3))
    m = size(Kxx, 1)
    fX = vec(mean(Kxx; dims = 1)) - mean(Kxy; dims = 2)
    fY = vec(mean(Kxy; dims = 1)) - vec(mean(Kyy; dims = 1))
    phist = plot(; title = "mmd = $(m * (mean(fX) - mean(fY)))")
    density!(phist, m .* fY; l = (4, :blue), label = "true data: m*fY")
    density!(phist, m .* fX; l = (4, :red),  label = "simulated: m*fX")

    return phist
end

####
#### Kernel bandwidth opt
####

function mmd_bandwidth_bruteopt(sampleX, sampleY, bounds; nsigma = 100, nevals = 100)
    # work = mmd_work(sampleX(), sampleY())
    function f(logσ)
        σ = exp(logσ)
        γ = inv(2*σ^2)
        k = Δ -> exp(-γ*Δ)
        X, Y = sampleX(), sampleY()
        mmds = [mmd(k, sampleX(), sampleY()) for _ in 1:nevals]
        MMDsq = mean(mmds)
        MMDvar = var(mmds)
        σ = √MMDvar
        MMDsq / σ
    end
    logσ = range(bounds[1], bounds[2]; length = nsigma)
    return logσ, [f(logσ) for logσ in logσ]
end

function mmd_bandwidth_optfun(logσ::Real, X, Y)
    m = size(X,2)
    σ = exp(logσ)
    γ = inv(2σ^2)
    k(Δ) = exp(-γ*Δ)
    MMDsq  = m * mmd(k, X, Y)
    MMDvar = m^2 * mmdvar(k, X, Y)
    ϵ = eps(typeof(MMDvar))
    t = MMDsq / √max(MMDvar, ϵ) # avoid division by zero/negative
    return t
end
∇mmd_bandwidth_optfun(logσ::Real, args...; kwargs...) = ForwardDiff.derivative(logσ -> mmd_bandwidth_optfun(logσ, args...; kwargs...), logσ)

function mmd_flux_bandwidth_optfun(logσ::AbstractArray, X, Y)
    # Avoiding div by zero/negative:
    #   m^2 * MMDvar >= ϵ  -->  m * MMDσ >= √ϵ
    MMDsq, MMDvar = mmd_and_mmdvar_flux(logσ, X, Y)
    m = size(X,2)
    ϵ = eps(typeof(MMDvar))
    t = m*MMDsq / √max(m^2*MMDvar, ϵ)
    return t
end
function ∇mmd_flux_bandwidth_optfun(logσ::AbstractArray, args...; kwargs...)
    if length(logσ) <= 16
        ForwardDiff.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)
    else
        Zygote.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)[1]
    end
end
∇mmd_flux_bandwidth_optfun_fwddiff(logσ::AbstractArray, args...; kwargs...) = ForwardDiff.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)
∇mmd_flux_bandwidth_optfun_zygote(logσ::AbstractArray, args...; kwargs...) = Zygote.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)[1]

#= (∇)mmd_(flux_)bandwidth_optfun speed + consistency testing
for m in [32,64,128,256,512], nsigma in [16], a in [2.0]
    logsigma = rand(nsigma)
    X, Y = randn(2,m), a.*randn(2,m)
    
    if nsigma == 1
        t1 = mmd_bandwidth_optfun(logsigma, X, Y)
        t2 = mmd_flux_bandwidth_optfun(logsigma, X, Y)
        # @show t1, t2
        # @show t1-t2
        @assert t1≈t2
    end

    g1 = nsigma == 1 ? ∇mmd_bandwidth_optfun(logsigma, X, Y) : nothing
    g2 = ∇mmd_flux_bandwidth_optfun_fwddiff(logsigma, X, Y)
    g3 = ∇mmd_flux_bandwidth_optfun_zygote(logsigma, X, Y)
    # @show g1, g2, g3
    # @show g1-g2, g2-g3
    @assert (nsigma != 1 || g1≈g2) && g2≈g3

    # @btime mmd_bandwidth_optfun($logsigma, $X, $Y)
    # @btime mmd_flux_bandwidth_optfun($logsigma, $X, $Y)

    @show m, nsigma
    (nsigma == 1) && @btime ∇mmd_bandwidth_optfun($logsigma, $X, $Y)
    @btime ∇mmd_flux_bandwidth_optfun_fwddiff($logsigma, $X, $Y)
    @btime ∇mmd_flux_bandwidth_optfun_zygote($logsigma, $X, $Y)
end
=#

####
#### Direct data samplers from prior derived from MLE fitted signals
####

function make_mle_data_samplers(
        imagepath,
        thetaspath;
        ntheta::Int,
        plothist = false,
    )

    # Set random seed for consistent test/train sets
    rng = Random.seed!(0)

    # Load + preprocess results
    results = deepcopy(BSON.load(thetaspath)["results"])
    #filter!(r -> r.T2short <= 100 && r.loss <= -200 && r.dT2 <= 500, results) # throw out ~2% of poor fits on mlefit-v1
    filter!(r -> r.T2short <= 100 && r.loss <= -230, results) # throw out ~4% of poor fits on mlefit-v2
    thetas = permutedims(convert(Matrix{Float64}, results[:, [:alpha, :T2short, :dT2, :Ashort]])) # convert to ntheta x nSamples Matrix

    # Plot prior distribution histograms
    plothist && display(plot(mapreduce(vcat, [:alpha, :T2short, :dT2, :Ashort, :logsigma, :loss]; init = Any[]) do c
        histogram(results[!,c]; lab = c, nbins = 75)
    end...))

    # Load image, keeping signals which correspond to thetas
    image = DECAES.load_image(imagepath) # load 4D MatrixSize x nTE image
    Y = convert(Matrix{Float64}, permutedims(image[CartesianIndex.(results[!, :index]), :])) # convert to nTE x nSamples Matrix
    Y ./= sum(Y; dims = 1) # Normalize signals to unit sum

    # Forward simulation params
    signal_work = signal_model_work(Float64; nTE = 48)
    signal_fun(θ::AbstractMatrix{Float64}, noise::Union{AbstractVector{Float64}, Nothing} = nothing) = signal_model!(signal_work, θ, noise; TE = 8e-3)

    # Generate data samplers
    itrain =                   1 : 2*(size(Y,2)÷4)
    itest  = 2*(size(Y,2)÷4) + 1 : 3*(size(Y,2)÷4)
    ival   = 3*(size(Y,2)÷4) + 1 : size(Y,2)

    # True data (Y) samplers
    Ytrain, Ytest, Yval = Y[:,itrain], Y[:,itest], Y[:,ival]
    sampleY = function(batchsize; dataset = :train)
        dataset == :train ? (batchsize === nothing ? Ytrain : sample_columns(Ytrain, batchsize)) :
        dataset == :test  ? (batchsize === nothing ? Ytest  : sample_columns(Ytest,  batchsize)) :
        dataset == :val   ? (batchsize === nothing ? Yval   : sample_columns(Yval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Fit parameters (θ) samplers
    θtrain, θtest, θval = thetas[:,itrain], thetas[:,itest], thetas[:,ival]
    sampleθ = function(batchsize; dataset = :train)
        dataset == :train ? (batchsize === nothing ? θtrain : sample_columns(θtrain, batchsize)) :
        dataset == :test  ? (batchsize === nothing ? θtest  : sample_columns(θtest,  batchsize)) :
        dataset == :val   ? (batchsize === nothing ? θval   : sample_columns(θval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers, possibly adding noise via `args...`
    _sampleX_model = function(batchsize, args...; kwargs...)
        signal_fun(sampleθ(batchsize; kwargs...), args...)
    end

    # Direct model data (X) samplers with no noise (precomputed)
    Xtrain = _sampleX_model(nothing; dataset = :train)
    Xtest  = _sampleX_model(nothing; dataset = :test)
    Xval   = _sampleX_model(nothing; dataset = :val)
    _sampleX_direct = function(batchsize; dataset = :train)
        dataset == :train ? (batchsize === nothing ? Xtrain : sample_columns(Xtrain, batchsize)) :
        dataset == :test  ? (batchsize === nothing ? Xtest  : sample_columns(Xtest,  batchsize)) :
        dataset == :val   ? (batchsize === nothing ? Xval   : sample_columns(Xval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers, possibly adding noise via `args...`
    sampleX = function(batchsize, args...; kwargs...)
        if batchsize === nothing && length(args) == 0
            _sampleX_direct(batchsize; kwargs...)
        else
            _sampleX_model(batchsize, args...; kwargs...)
        end
    end

    # Reset random seed
    Random.seed!(rng)

    return sampleX, sampleY, sampleθ
end

####
#### GMM data samplers from learned prior
####

function make_gmm_data_samplers(
        image;
        ntheta::Int
    )

    # Set random seed for consistent test/train sets
    rng = Random.seed!(0)

    function read_results(results_dir)
        results = DataFrame(refcon = Float64[], alpha = Float64[], T2short = Float64[], dT2 = Float64[], Ashort = Float64[])
        for (root, dirs, files) in walkdir(results_dir)
            for file in files
                if file == "bbsignalfit_results.mat"
                    θ = DECAES.MAT.matread(joinpath(root, file))["thetas"]'
                    df = similar(results, size(θ,1))
                    df[!,:] .= θ
                    append!(results, df)
                end
            end
        end
        results.T2long = results.T2short .+ results.dT2
        results.Along  = 1 .- results.Ashort
        return results
    end

    # Read in simulation results from file
    results_dir = "/scratch/st-arausch-1/jcd1994/MMD-Learning/sigfit-v5"
    results = read_results(results_dir)

    # Transform data by shifting data to γ * [-0.5, 0.5] and applying tanh.
    # This makes data more smoothly centred around zero, with
    #   γ = 2*tanh(3) ≈ 1.99
    # sending boundary points to approx. +/- 3
    f(x,a,b) = atanh((x - ((a+b)/2)) * (1.99 / (b-a)))
    g(y,a,b) = ((a+b)/2) + tanh(y) * ((b-a) / 1.99)
    f(x,t::NTuple{2}) = f(x,t...)
    g(x,t::NTuple{2}) = g(x,t...)
    trans!(fun, df, bounds) = (foreach(j -> df[!,j] .= fun.(df[!,j], Ref(bounds[j])), 1:ncol(df)); return df)

    thetas = results[:, [:refcon, :alpha, :T2short, :T2long, :Ashort]]
    filter!(row -> !(row.Ashort ≈ 1) && 10 <= row.T2short <= 100 && row.T2long <= 500, thetas)
    bounds = map(extrema, eachcol(thetas))
    thetas_trans = trans!(f, copy(thetas), bounds)
    bounds_trans = map(extrema, eachcol(thetas_trans))

    # Fit GMM to data
    gmm = GMM(32, Matrix(thetas_trans); method = :kmeans, kind = :full, nInit = 1000, nIter = 100, nFinal = 100)

    # Generate data samplers
    Y = signal_data(image)
    Y = Y[:, randperm(size(Y,2))] # shuffle data
    itrain = 1:2*(size(Y,2)÷4)
    itest  = itrain[end]+1:3*(size(Y,2)÷4)
    ival   = itest[end]+1:size(Y,2)

    Ytrain, Ytest, Yval = Y[:,itrain], Y[:,itest], Y[:,ival]
    sampleY = function(batchsize; dataset = :train)
        dataset == :train ? (batchsize === nothing ? Ytrain : sample_columns(Ytrain, batchsize)) :
        dataset == :test  ? (batchsize === nothing ? Ytest  : sample_columns(Ytest,  batchsize)) :
        dataset == :val   ? (batchsize === nothing ? Yval   : sample_columns(Yval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    sampleθ = function(batchsize)
        θ = zeros(ntheta, 0)
        while size(θ, 2) < batchsize
            draws_trans = rand(gmm, batchsize)
            idx = [all(j -> bounds_trans[j][1] <= draws_trans[i,j] <= bounds_trans[j][2], 1:size(draws_trans,2)) for i in 1:size(draws_trans,1)]
            draws_trans = draws_trans[idx, :]
            θnew = [g.(row, bounds) for row in eachrow(draws_trans)]
            if !isempty(θnew)
                θ = hcat(θ, reduce(hcat, θnew))
            end
        end
        θ = θ[:, 1:batchsize]
        return θ
    end

    signal_work = signal_model_work(Float64; nTE = 48)
    signal_fun(θ::AbstractMatrix{Float64}, noise::Union{AbstractVector{Float64}, Nothing} = nothing) = signal_model!(signal_work, θ, noise; TE = 8e-3)

    # Args is variadic such that noise may be passed to X sampler
    sampleX = function(batchsize, args...; kwargs...)
        signal_fun(sampleθ(batchsize; kwargs...), args...)
    end

    # Reset random seed
    Random.seed!(rng)

    return sampleX, sampleY, sampleθ
end

####
#### Toy problem samplers
####

function toy_signal_model(
        theta::AbstractVecOrMat,
        epsilon = nothing,
        power = 2;
        nsamples = 128,
    )
    @assert size(theta, 1) == 5
    freq   = theta[1:1,:]
    phase  = theta[2:2,:]
    offset = theta[3:3,:]
    amp    = theta[4:4,:]
    tconst = theta[5:5,:]
    ts = 0:nsamples-1
    y = @. (offset + amp * abs(sin(2*(pi*freq*ts) - phase))^power) * exp(-ts/tconst)
    if !isnothing(epsilon)
        y = @. rand(Rician(y, epsilon))
    end
    return y
end
toy_signal_model(n::Int, args...; kwargs...) = toy_signal_model(toy_theta_sampler(n), args...; kwargs...)

toy_theta_bounds(T = Float64) = map(x -> T.(x), [(1/64, 1/32), (0.0, pi/2), (0.25, 0.5), (0.1, 0.25), (16.0, 128.0)])

function toy_theta_sampler(n::Int = 1)
    freq   = rand(Uniform(1/64,  1/32), n)
    phase  = rand(Uniform( 0.0,  pi/2), n)
    offset = rand(Uniform(0.25,   0.5), n)
    amp    = rand(Uniform( 0.1,  0.25), n)
    tconst = rand(Uniform(16.0, 128.0), n)
    return permutedims(hcat(freq, phase, offset, amp, tconst))
end

function toy_theta_error(theta, thetahat)
    return abs.((theta .- thetahat)) ./ [1/32 - 1/64, pi/2 - 0.0, 0.5 - 0.25, 0.25 - 0.1, 128.0 - 16.0]
end

function make_toy_samplers(;
        input_noise = false,
        epsilon = nothing,
        power = 4.0,
        ntrain = 1_000,
        ntest = ntrain,
        nval = ntrain,
    )

    # Set random seed for consistent test/train/val sets
    rng = Random.seed!(0)

    # Sample parameter prior space
    sampleθ = toy_theta_sampler

    # Sample (incorrect) model
    sampleX = input_noise ?
        (batchsize, noise) -> toy_signal_model(batchsize, noise, power) :
        (batchsize) -> toy_signal_model(batchsize, nothing, power)

    # Sample data
    _sampleY(batchsize) = toy_signal_model(batchsize, epsilon, 2)
    Ytrain, Ytest, Yval = _sampleY(ntrain), _sampleY(ntest), _sampleY(nval)
    sampleY = function(batchsize; dataset = :train)
        dataset == :train ? (batchsize === nothing ? Ytrain : _sampleY(batchsize)) :
        dataset == :test  ? (batchsize === nothing ? Ytest  : _sampleY(batchsize)) :
        dataset == :val   ? (batchsize === nothing ? Yval   : _sampleY(batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Reset random seed
    Random.seed!(rng)

    return sampleX, sampleY, sampleθ
end

####
#### Maximum likelihood estimation inference
####

function toy_theta_loglikelihood_inference(
        y::AbstractVector,
        initial_guess = nothing,
        model = x -> (x, zero(x)),
        signal_fun = θ -> toy_signal_model(θ, nothing, 4);
        bounds = toy_theta_bounds(),
        objective = :mle,
        bbopt_kwargs = Dict(:MaxTime => 1.0),
    )

    # Deterministic loss function, suitable for Optim
    function mle_loss(θ)
        ȳhat, ϵhat = model(signal_fun(θ))
        return sum(@. -logpdf(Rician(ȳhat, ϵhat; check_args = false), y))
    end

    # Stochastic loss function, only suitable for BlackBoxOptim
    function rmse_loss(θ)
        ȳhat, ϵhat = model(signal_fun(θ))
        yhat = @. rand(Rician(ȳhat, ϵhat; check_args = false))
        return sqrt(mean(abs2, y .- yhat))
    end

    loss = objective === :mle ? mle_loss : rmse_loss

    bbres = nothing
    if objective !== :mle || (objective === :mle && isnothing(initial_guess))
        bbres = bboptimize(loss;
            SearchRange = bounds,
            TraceMode = :silent,
            bbopt_kwargs...
        )
    end

    optres = nothing
    if objective === :mle
        θ0 = isnothing(initial_guess) ? BlackBoxOptim.best_candidate(bbres) : initial_guess
        lo = (x->x[1]).(bounds)
        hi = (x->x[2]).(bounds)
        # dfc = Optim.TwiceDifferentiableConstraints(lo, hi)
        # df = Optim.TwiceDifferentiable(loss, θ0; autodiff = :forward)
        # optres = Optim.optimize(df, dfc, θ0, Optim.IPNewton())
        df = Optim.OnceDifferentiable(loss, θ0; autodiff = :forward)
        optres = Optim.optimize(df, lo, hi, θ0, Optim.Fminbox(Optim.LBFGS()))
        # optres = Optim.optimize(df, lo, hi, θ0, Optim.Fminbox(Optim.BFGS()))
    end

    return @ntuple(bbres, optres)
end
function toy_theta_loglikelihood_inference(Y::AbstractMatrix, θ0::Union{<:AbstractMatrix, Nothing} = nothing, args...; kwargs...)
    _args = [deepcopy(args) for _ in 1:Threads.nthreads()]
    _kwargs = [deepcopy(kwargs) for _ in 1:Threads.nthreads()]
    ThreadPools.qmap(1:size(Y,2)) do j # map(1:size(Y,2)) do j
        tid = Threads.threadid()
        initial_guess = !isnothing(θ0) ? θ0[:,j] : nothing
        toy_theta_loglikelihood_inference(Y[:,j], initial_guess, _args[tid]...; _kwargs[tid]...)
    end
end

#=
for _ in 1:1
    noise_level = 1e-2
    θ = toy_theta_sampler(1);
    x = toy_signal_model(θ, nothing, 4);
    y = toy_signal_model(θ, nothing, 2);
    xϵ = toy_signal_model(θ, noise_level, 4);
    yϵ = toy_signal_model(θ, noise_level, 2);

    m = x -> ((dx, ϵ) = get_correction_and_noise(x); return (abs.(x.+dx), ϵ));

    @time bbres1, _ = toy_theta_loglikelihood_inference(yϵ, nothing, m; objective = :rmse)[1];
    θhat1 = BlackBoxOptim.best_candidate(bbres1);
    xhat1 = toy_signal_model(θhat1, nothing, 4);
    dxhat1, ϵhat1 = get_correction_and_noise(xhat1);
    yhat1 = get_corrected_signal(xhat1, dxhat1, ϵhat1);

    @time bbres2, optres2 = toy_theta_loglikelihood_inference(yϵ, nothing, m; objective = :mle)[1];
    θhat2 = Optim.minimizer(optres2); #BlackBoxOptim.best_candidate(bbres2);
    xhat2 = toy_signal_model(θhat2, nothing, 4);
    dxhat2, ϵhat2 = get_correction_and_noise(xhat2);
    yhat2 = get_corrected_signal(xhat2, dxhat2, ϵhat2);

    p1 = plot([y[:,1] x[:,1]]; label = ["Yθ" "Xθ"], line = (2,));
    p2 = plot([yϵ[:,1] xϵ[:,1]]; label = ["Yθϵ" "Xθϵ"], line = (2,));
    p3 = plot([yϵ[:,1] yhat1]; label = ["Yθϵ" "Ȳθϵ₁"], line = (2,));
    p4 = plot([yϵ[:,1] yhat2]; label = ["Yθϵ" "Ȳθϵ₂"], line = (2,));
    plot(p1,p2,p3,p4) |> display;

    @show toy_theta_error(θ[:,1], θhat1)';
    @show toy_theta_error(θ[:,1], θhat2)';
    @show √mean(abs2, y[:,1] .- (xhat1 .+ dxhat1));
    @show √mean(abs2, y[:,1] .- (xhat2 .+ dxhat2));
    @show √mean([mean(abs2, yϵ[:,1] .- get_corrected_signal(xhat1, dxhat1, ϵhat1)) for _ in 1:1000]);
    @show √mean([mean(abs2, yϵ[:,1] .- get_corrected_signal(xhat2, dxhat2, ϵhat2)) for _ in 1:1000]);
end;
=#

####
#### Toy problem MCMC inference
####

#=
Turing.@model toy_model_rician_noise(
        y,
        correction_and_noise,
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
    yhat, ϵhat = correction_and_noise(x)

    # Model noise as Rician
    for i in 1:length(y)
        # ν, σ = x[i], epsilon
        ν, σ = yhat[i], ϵhat[i]
        y[i] ~ Rician(ν, σ; check_args = false)
    end
end
=#

function toy_theta_mcmc_inference(
        y::AbstractVector,
        correction_and_noise,
        callback = (y, chain) -> true,
    )
    model = function (x)
        xhat, ϵhat = correction_and_noise(x)
        yhat = @. rand(Rician(xhat, ϵhat; check_args = false))
        return yhat
    end
    res = toy_theta_loglikelihood_inference(y, nothing, model)
    theta0 = best_candidate(res)
    while true
        chain = sample(toy_model_rician_noise(y, correction_and_noise), NUTS(), 1000; verbose = true, init_theta = theta0)
        # chain = psample(toy_model_rician_noise(y, correction_and_noise), NUTS(), 1000, 3; verbose = true, init_theta = theta0)
        callback(y, chain) && return chain
    end
end
function toy_theta_mcmc_inference(Y::AbstractMatrix, args...; kwargs...)
    ThreadPools.qmap(1:size(Y,2)) do j
        toy_theta_mcmc_inference(Y[:,j], args...; kwargs...)
    end
end

function find_cutoff(x; initfrac = 0.25, pthresh = 1e-4)
    cutoff = clamp(round(Int, initfrac * length(x)), 2, length(x))
    for i = cutoff+1:length(x)
        mu = mean(x[1:i-1])
        sig = std(x[1:i-1])
        p = ccdf(Normal(mu, sig), x[i])
        (p < pthresh) && break
        cutoff += 1
    end
    return cutoff
end

#=
for _ in 1:1
    correction_and_noise = let _model = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/2020-02-20T15:43:48.506/best-model.bson")["model"]) #deepcopy(model)
        function(x)
            out = _model(x)
            dx, logϵ = out[1:end÷2], out[end÷2+1:end]
            return abs.(x .+ dx), exp.(logϵ)
        end
    end
    signal_model = function(θhat)
        x = toy_signal_model(θhat, nothing, 4)
        xhat, ϵhat = correction_and_noise(x)
        # zR = ϵhat .* randn(size(x)...)
        # zI = ϵhat .* randn(size(x)...)
        # yhat = @. sqrt((xhat + zR)^2 + zI^2)
        yhat = @. rand(Rician(xhat, ϵhat))
    end
    fitresults = function(y, c)
        θhat = map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst])
        # ϵhat = 10^map(k -> mean(c[k])[1,:mean], [:logeps])[1]
        # yhat, ϵhat = correction_and_noise(toy_signal_model(θhat, nothing, 4))
        yhat = signal_model(θhat)
        yerr = sqrt(mean(abs2, y - yhat))
        @ntuple(θhat, yhat, yerr)
    end
    plotresults = function(y, c)
        @unpack θhat, yhat, yerr = fitresults(y, c)
        display(plot(c))
        display(plot([y yhat]))
        return nothing
        # return plot(c) #|> display
        # return plot([y yhat]) #|> display
    end

    # θ = [freq, phase, offset, amp, tconst]
    # Random.seed!(0);
    noise_level = 1e-2;
    θ = toy_theta_sampler(16);
    Y = toy_signal_model(θ, noise_level, 2);

    # ThreadPools.qmap(_ -> Random.seed!(0), 1:Threads.nthreads());
    # @time cs = toy_theta_mcmc_inference(Y, correction_and_noise);
    # res = map(j -> fitresults(Y[:,j], cs[j]), 1:size(Y,2))
    # ps = map(j -> plotresults(Y[:,j], cs[j]), 1:size(Y,2))
    # θhat = reduce(hcat, map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst]) for c in cs)
    # Yerr = sort(getfield.(res, :yerr))

    @time bbres = toy_theta_loglikelihood_inference(Y, nothing, signal_model);
    Yerr = sort(best_fitness.(bbres))
    θhat = best_candidate.(bbres)
    Yhat = signal_model.(θhat)
    θhat = reduce(hcat, θhat)
    Yhat = reduce(hcat, Yhat)
    map(j -> display(plot([Y[:,j] Yhat[:,j]])), 1:size(Y,2))

    let
        p = plot()
        sticks!(p, Yerr; m = (:circle,4), lab = "Yerr")
        # sticks!(p, [0; diff(Yerr)]; m = (:circle,4), lab = "dYerr")
        hline!(p, [2noise_level]; lab = "2ϵ")
        vline!(p, [find_cutoff(Yerr; pthresh = 1e-4)]; lab = "cutoff", line = (:black, :dash))
        display(p)
    end

    display(θhat)
    display(θ)
    display((θ.-θhat)./θ)
end
=#

#=
for _ in 1:100
    let seed = rand(0:1000_000)
        rng = Random.seed!(seed)
        p = plot();
        Random.seed!(seed); plot!(toy_signal_model(3, nothing, 2); ylim = (0, 1.5));
        Random.seed!(seed); plot!(toy_signal_model(3, nothing, 2.5); ylim = (0, 1.5));
        display(p);
        Random.seed!(rng);
    end;
end;
=#

nothing
