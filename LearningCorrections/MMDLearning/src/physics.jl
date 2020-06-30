####
#### Rician correctors
####

abstract type RicianCorrector end

# G : 𝐑^n -> 𝐑^2n mapping X ∈ 𝐑^n ⟶ δ,logϵ ∈ 𝐑^n concatenated as [δ; logϵ]
@with_kw struct VectorRicianCorrector{Gtype} <: RicianCorrector
    G::Gtype
end

# G : 𝐑^n -> 𝐑^n mapping X ∈ 𝐑^n ⟶ δ ∈ 𝐑^n with fixed noise ϵ0 ∈ 𝐑, or ϵ0 ∈ 𝐑^n
@with_kw struct FixedNoiseVectorRicianCorrector{Gtype,T} <: RicianCorrector
    G::Gtype
    ϵ0::T
end

# Concrete methods to extract δ and ϵ
function correction_and_noiselevel(G::VectorRicianCorrector, X)
    δ_logϵ = G.G(X)
    δ_logϵ[1:end÷2, :], exp.(δ_logϵ[end÷2+1:end, :])
end
correction_and_noiselevel(G::FixedNoiseVectorRicianCorrector, X) = G.G(X), ϵ0

# Derived convenience functions
correction(G::RicianCorrector, X) = correction_and_noiselevel(G, X)[1]
noiselevel(G::RicianCorrector, X) = correction_and_noiselevel(G, X)[2]
noise_instance(G::RicianCorrector, X, ϵ) = ϵ .* randn(eltype(X), size(X)...)
noise_instance(G::RicianCorrector, X) = noise_instance(G, X, noiselevel(G, X))
corrected_signal_instance(G::RicianCorrector, X) = corrected_signal_instance(G, X, correction_and_noiselevel(G, X)...)
corrected_signal_instance(G::RicianCorrector, X, δ, ϵ) = corrected_signal_instance(G, abs.(X .+ δ), ϵ)
function corrected_signal_instance(G::RicianCorrector, X, ϵ)
    ϵR = noise_instance(G, X, ϵ)
    ϵI = noise_instance(G, X, ϵ)
    Xϵ = @. sqrt((X + ϵR)^2 + ϵI^2)
    return Xϵ
end
function rician_params(G::RicianCorrector, X)
    δ, ϵ = correction_and_noiselevel(G, X)
    ν, σ = abs.(X .+ δ), ϵ
    return ν, σ
end

####
#### Physics model interface
####

abstract type PhysicsModel end

struct ClosedForm{P<:PhysicsModel}
    p::P
end

# Abstract interface
hasclosedform(p::PhysicsModel) = false # fallback
physicsmodel(p::PhysicsModel) = p
physicsmodel(c::ClosedForm) = c.p
θbounds(p::PhysicsModel) = tuple.(θlower(p), θupper(p))
θbounds(c::ClosedForm) = θbounds(physicsmodel(c))
ntheta(c::ClosedForm) = ntheta(physicsmodel(c))
nsignal(c::ClosedForm) = nsignal(physicsmodel(c))
function epsilon end

####
#### Toy problem
####

@with_kw struct ToyModel{T} <: PhysicsModel
    ϵ0::T = T(0.01)
    Ytrain::Ref{Matrix{T}} = Ref(zeros(T,0,0))
    Ytest::Ref{Matrix{T}} = Ref(zeros(T,0,0))
    Yval::Ref{Matrix{T}} = Ref(zeros(T,0,0))
end
const ClosedFormToyModel{T} = ClosedForm{ToyModel{T}}
const MaybeClosedFormToyModel{T} = Union{ToyModel{T}, ClosedFormToyModel{T}}

ntheta(::ToyModel) = 5
nsignal(::ToyModel) = 128
hasclosedform(::ToyModel) = true
beta(::ToyModel) = 4
beta(::ClosedFormToyModel) = 2
epsilon(c::ClosedFormToyModel) = physicsmodel(c).ϵ0

θlabels(::ToyModel) = ["freq", "phase", "offset", "amp", "tconst"]
θlower(::ToyModel{T}) where {T} = [1/T(64),   T(0), 1/T(4), 1/T(10),  T(16)]
θupper(::ToyModel{T}) where {T} = [1/T(32), T(π)/2, 1/T(2),  1/T(4), T(128)]
θerror(p::ToyModel, theta, thetahat) = abs.((theta .- thetahat)) ./ (θupper(p) .- θlower(p))

function initialize!(p::ToyModel; ntrain::Int, ntest::Int = ntrain, nval::Int = ntrain)
    rng = Random.seed!(0)
    p.Ytrain[] = sampleX(ClosedForm(p), ntrain, p.ϵ0; dataset = :train)
    p.Ytest[]  = sampleX(ClosedForm(p), ntest, p.ϵ0; dataset = :test)
    p.Yval[]   = sampleX(ClosedForm(p), nval, p.ϵ0; dataset = :val)
    Random.seed!(rng)
    return p
end

sampleθ(p::ToyModel, n::Union{Int, Symbol}; dataset::Symbol) = permutedims(reduce(hcat, rand.(Uniform.(θlower(p), θupper(p)), n)))

sampleX(p::MaybeClosedFormToyModel, n::Union{Int, Symbol}, epsilon = nothing; dataset::Symbol) = sampleX(p, sampleθ(physicsmodel(p), n; dataset = dataset), epsilon)
sampleX(p::MaybeClosedFormToyModel, theta, epsilon = nothing) = signal_model(p, theta, epsilon)

function sampleY(p::ToyModel, n::Union{Int, Symbol}; dataset::Symbol)
    dataset === :train ? (n === :all ? p.Ytrain[] : sample_columns(p.Ytrain[], n)) :
    dataset === :test  ? (n === :all ? p.Ytest[]  : sample_columns(p.Ytest[], n)) :
    dataset === :val   ? (n === :all ? p.Yval[]   : sample_columns(p.Yval[], n)) :
    error("dataset must be :train, :test, or :val")
end

function _signal_model(
        theta::AbstractVecOrMat,
        epsilon,
        nsamples::Int,
        beta::Int,
    )
    freq, phase, offset, amp, tconst = theta[1:1,:], theta[2:2,:], theta[3:3,:], theta[4:4,:], theta[5:5,:]
    t = 0:nsamples-1
    y = @. (offset + amp * sin(2*(pi*freq)*t - phase)^beta) * exp(-t/tconst)
    if !isnothing(epsilon)
        ϵR = epsilon .* randn(eltype(theta), nsamples, size(theta, 2))
        ϵI = epsilon .* randn(eltype(theta), nsamples, size(theta, 2))
        y = @. sqrt((y + ϵR)^2 + ϵI^2)
    end
    return y
end
signal_model(p::MaybeClosedFormToyModel, theta::AbstractVecOrMat, epsilon = nothing) = _signal_model(theta, epsilon, nsignal(p), beta(p))

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
        # theta_labels = ["alpha", "T2short", "dT2", "Ashort"]
        theta_lb = T[ 50.0,    8.0,    8.0, 0.0]
        theta_ub = T[180.0, 1000.0, 1000.0, 1.0]
    elseif ntheta == 5
        # theta_labels = ["alpha", "T2short", "dT2", "Ashort", "Along"]
        theta_lb = T[ 50.0,    8.0,    8.0, 0.0, 0.0]
        theta_ub = T[180.0, 1000.0, 1000.0, 1.0, 1.0]
    else
        error("Number of labels must be 4 or 5")
    end
    theta_bd = collect(zip(theta_lb, theta_ub))
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
        TE,
        normalize::Bool = true,
    ) where {T}
    @unpack epg_work, signal, real_noise, imag_noise = work
    # alpha, T2short, dT2, Ashort = θ[1], θ[2]/1000, θ[3]/1000, θ[4]
    alpha, T2short, dT2, Ashort, Along = θ[1], θ[2]/1000, θ[3]/1000, θ[4], θ[5]
    refcon  = T(180.0)
    T2long  = T2short + dT2
    #Along  = 1-Ashort
    T1short = T(1000.0)/1000
    T1long  = T(1000.0)/1000

    signal  .= Ashort .* DECAES.EPGdecaycurve!(epg_work, alpha, T(TE), T2short, T1short, refcon) # short component
    signal .+= Along  .* DECAES.EPGdecaycurve!(epg_work, alpha, T(TE), T2long,  T1long,  refcon) # long component
    normalize && (signal ./= sum(signal))

    # Add noise to "real" and "imag" channels in quadrature
    if !isnothing(ϵ)
        if eltype(ϵ) === eltype(θ)
            noise_model!(real_noise, signal, ϵ) # populate real_noise
            noise_model!(imag_noise, signal, ϵ) # populate imag_noise
            signal .= sqrt.((signal .+ real_noise).^2 .+ imag_noise.^2)
            normalize && (signal ./= sum(signal))
        else
            # Add forward-differentiable noise to "real" and "imag" channels in quadrature
            signal = sqrt.((signal .+ noise_model(signal, ϵ)).^2 .+ noise_model(signal, ϵ).^2)
            normalize && (signal ./= sum(signal))
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

signal_model(θ::AbstractVecOrMat{T}, ϵ::Union{AbstractVector, Nothing} = nothing; nTE::Int, TE, kwargs...) where {T} = signal_model!(signal_model_work(T; nTE = nTE), θ, ϵ; TE = T(TE), kwargs...)

function mutate_signal(Y::AbstractVecOrMat; meanmutations::Int = 0)
    if meanmutations <= 0
        return Y
    end
    nrow = size(Y, 1)
    p = meanmutations / nrow
    return Y .* (rand(size(Y)...) .> p)
end

####
#### Direct data samplers from prior derived from MLE fitted signals
####

function make_mle_data_samplers(
        imagepath,
        thetaspath;
        ntheta::Int,
        plothist = false,
        padtrain = false,
        normalizesignals = true,
        filteroutliers = false,
    )
    @assert !(padtrain && !normalizesignals) "unnormalized padded training data is not implemented"

    # Set random seed for consistent test/train sets
    rng = Random.seed!(0)

    # Load + preprocess fit results (~25% of voxels dropped)
    fits = deepcopy(BSON.load(thetaspath)["results"])

    if filteroutliers
        println("before filter: $(nrow(fits))")
        filter!(r -> !(999.99 <= r.dT2 && 999.99 <= r.T2short), fits) # drop boundary failures
        filter!(r -> r.dT2 <= 999.99, fits) # drop boundary failures
        filter!(r -> r.T2short <= 100, fits) # drop long T2short (very few points)
        filter!(r -> 8.01 <= r.T2short, fits) # drop boundary failures
        filter!(r -> 8.01 <= r.dT2, fits) # drop boundary failures
        if ntheta == 5
            filter!(r -> 0.005 <= r.Ashort <= 0.15, fits) # drop outlier fits (very few points)
            filter!(r -> 0.005 <= r.Along <= 0.15, fits) # drop outlier fits (very few points)
        end
        filter!(r -> r.loss <= -250, fits) # drop poor fits (very few points)
        println("after filter:  $(nrow(fits))")
    end

    # Shuffle data + collect thetas
    fits = fits[shuffle(MersenneTwister(0), 1:nrow(fits)), :]
    thetas = ntheta == 4 ? # Create ntheta x nSamples matrix
        permutedims(convert(Matrix{Float64}, fits[:, [:alpha, :T2short, :dT2, :Ashort]])) :
        permutedims(convert(Matrix{Float64}, fits[:, [:alpha, :T2short, :dT2, :Ashort, :Along]]))

    # Load image, keeping signals which correspond to thetas
    image = DECAES.load_image(imagepath) # load 4D MatrixSize x nTE image
    Y = convert(Matrix{Float64}, permutedims(image[CartesianIndex.(fits[!, :index]), :])) # convert to nTE x nSamples Matrix

    if normalizesignals
        # Normalize signals individually such that each signal has unit sum
        Y ./= sum(Y; dims = 1)
    else
        # Don't normalize scales individually, but nevertheless scale the signals uniformly down to avoid numerical difficulties
        Y ./= 1e6
        Ysum = sum(Y; dims = 1)

        # Scale thetas and fit results for using unnormalized Y
        thetas[4:4, :] .*= Ysum # scale Ashort
        fits.Ashort .*= vec(Ysum)
        if ntheta == 5
            thetas[5:5, :] .*= Ysum # scale Along
            fits.Along .*= vec(Ysum)
        end
        fits.logsigma .= log.(exp.(fits.logsigma) .* vec(Ysum)) # scale sigma from fit results
        fits.rmse .*= vec(Ysum) # scale rmse from fit results
        fits.loss .+= size(Y,1) .* log.(vec(Ysum)) # scale mle loss from fit results
    end

    # Forward simulation params
    signal_work = signal_model_work(Float64; nTE = 48)
    signal_fun(θ::AbstractMatrix{Float64}, noise::Union{AbstractVector{Float64}, Nothing} = nothing; kwargs...) =
        signal_model!(signal_work, θ, noise; TE = 8e-3, normalize = normalizesignals, kwargs...)

    # Pad training data with thetas sampled uniformly randomly over the prior space
    local θtrain_pad
    if padtrain
        @assert normalizesignals "unnormalized padded training data is not implemented"
        θ_pad_lo, θ_pad_hi = minimum(thetas; dims = 2), maximum(thetas; dims = 2)
        θtrain_pad = θ_pad_lo .+ (θ_pad_hi .- θ_pad_lo) .* rand(MersenneTwister(0), ntheta, nrow(fits))
        Xtrain_pad = signal_fun(θtrain_pad; normalize = false)
        if ntheta == 5
            θtrain_pad[4:5, :] ./= sum(Xtrain_pad; dims = 1) # normalize Ashort, Along
            train_pad_filter   = map(Ashort -> 0.005 <= Ashort <= 0.15, θtrain_pad[4,:]) # drop outlier samples (very few points)
            train_pad_filter .&= map(Along  -> 0.005 <= Along  <= 0.15, θtrain_pad[5,:]) # drop outlier samples (very few points)
            θtrain_pad = θtrain_pad[:, train_pad_filter]
        end
        println("num padded:    $(size(θtrain_pad,2))")
    end

    # Plot prior distribution histograms
    if plothist
        theta_cols = ntheta == 4 ? [:alpha, :T2short, :dT2, :Ashort] : [:alpha, :T2short, :dT2, :Ashort, :Along]
        display(plot([histogram(fits[!,c]; lab = c, nbins = 75) for c in [theta_cols; :logsigma; :loss]]...))
    end

    # Generate data samplers
    itrain =                   1 : 2*(size(Y,2)÷4)
    itest  = 2*(size(Y,2)÷4) + 1 : 3*(size(Y,2)÷4)
    ival   = 3*(size(Y,2)÷4) + 1 : size(Y,2)

    # True data (Y) samplers
    Ytrain, Ytest, Yval = Y[:,itrain], Y[:,itest], Y[:,ival]
    function sampleY(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? Ytrain : sample_columns(Ytrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? Ytest  : sample_columns(Ytest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? Yval   : sample_columns(Yval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Fit parameters (θ) samplers
    θtrain, θtest, θval = thetas[:,itrain], thetas[:,itest], thetas[:,ival]
    if padtrain
        θtrain = hcat(θtrain, θtrain_pad)
        θtrain = θtrain[:,shuffle(MersenneTwister(0), 1:size(θtrain,2))] # mix training + padded thetas
    end
    function sampleθ(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? θtrain : sample_columns(θtrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? θtest  : sample_columns(θtest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? θval   : sample_columns(θval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers
    function _sampleX_model(batchsize; dataset = :train, kwargs...)
        signal_fun(sampleθ(batchsize; dataset = dataset); kwargs...)
    end

    # Direct model data (X) samplers
    Xtrain = _sampleX_model(nothing; dataset = :train)
    Xtest  = _sampleX_model(nothing; dataset = :test)
    Xval   = _sampleX_model(nothing; dataset = :val)
    function _sampleX_direct(batchsize; dataset = :train)
        dataset === :train ? (batchsize === nothing ? Xtrain : sample_columns(Xtrain, batchsize)) :
        dataset === :test  ? (batchsize === nothing ? Xtest  : sample_columns(Xtest,  batchsize)) :
        dataset === :val   ? (batchsize === nothing ? Xval   : sample_columns(Xval,   batchsize)) :
        error("dataset must be :train, :test, or :val")
    end

    # Model data (X) samplers
    function sampleX(batchsize; kwargs...)
        if batchsize === nothing
            _sampleX_direct(batchsize; kwargs...)
        else
            _sampleX_model(batchsize; kwargs...)
        end
    end

    # Output train/test/val dataframe partitions
    fits_train, fits_test, fits_val = fits[itrain,:], fits[itest,:], fits[ival,:]

    # Reset random seed
    Random.seed!(rng)

    return @ntuple(sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val)
end

####
#### Maximum likelihood estimation inference
####

function signal_loglikelihood_inference(
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
        return -sum(logpdf.(Rician.(ȳhat, ϵhat), y))
    end

    # Stochastic loss function, only suitable for BlackBoxOptim
    function rmse_loss(θ)
        ȳhat, ϵhat = model(signal_fun(θ))
        yhat = rand.(Rician.(ȳhat, ϵhat))
        return sqrt(mean(abs2, y .- yhat))
    end

    loss = objective === :mle ? mle_loss : rmse_loss

    bbres = nothing
    if objective !== :mle || (objective === :mle && isnothing(initial_guess))
        bbres = BlackBoxOptim.bboptimize(loss;
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
function signal_loglikelihood_inference(Y::AbstractMatrix, θ0::Union{<:AbstractMatrix, Nothing} = nothing, args...; kwargs...)
    _args = [deepcopy(args) for _ in 1:Threads.nthreads()]
    _kwargs = [deepcopy(kwargs) for _ in 1:Threads.nthreads()]
    tasks = map(1:size(Y,2)) do j
        Threads.@spawn begin
            tid = Threads.threadid()
            initial_guess = !isnothing(θ0) ? θ0[:,j] : nothing
            signal_loglikelihood_inference(Y[:,j], initial_guess, _args[tid]...; _kwargs[tid]...)
        end
    end
    return map(Threads.fetch, tasks)
end

#=
for _ in 1:1
    noise_level = 1e-2
    θ = toy_theta_sampler(1);
    x = toy_signal_model(θ, nothing, 4);
    y = toy_signal_model(θ, nothing, 2);
    xϵ = toy_signal_model(θ, noise_level, 4);
    yϵ = toy_signal_model(θ, noise_level, 2);

    m = x -> ((dx, ϵ) = correction_and_noiselevel(x); return (abs.(x.+dx), ϵ));

    @time bbres1, _ = signal_loglikelihood_inference(yϵ, nothing, m; objective = :rmse)[1];
    θhat1 = BlackBoxOptim.best_candidate(bbres1);
    xhat1 = toy_signal_model(θhat1, nothing, 4);
    dxhat1, ϵhat1 = correction_and_noiselevel(xhat1);
    yhat1 = corrected_signal_instance(xhat1, dxhat1, ϵhat1);

    @time bbres2, optres2 = signal_loglikelihood_inference(yϵ, nothing, m; objective = :mle)[1];
    θhat2 = Optim.minimizer(optres2); #BlackBoxOptim.best_candidate(bbres2);
    xhat2 = toy_signal_model(θhat2, nothing, 4);
    dxhat2, ϵhat2 = correction_and_noiselevel(xhat2);
    yhat2 = corrected_signal_instance(xhat2, dxhat2, ϵhat2);

    p1 = plot([y[:,1] x[:,1]]; label = ["Yθ" "Xθ"], line = (2,));
    p2 = plot([yϵ[:,1] xϵ[:,1]]; label = ["Yθϵ" "Xθϵ"], line = (2,));
    p3 = plot([yϵ[:,1] yhat1]; label = ["Yθϵ" "Ȳθϵ₁"], line = (2,));
    p4 = plot([yϵ[:,1] yhat2]; label = ["Yθϵ" "Ȳθϵ₂"], line = (2,));
    plot(p1,p2,p3,p4) |> display;

    @show toy_theta_error(θ[:,1], θhat1)';
    @show toy_theta_error(θ[:,1], θhat2)';
    @show √mean(abs2, y[:,1] .- (xhat1 .+ dxhat1));
    @show √mean(abs2, y[:,1] .- (xhat2 .+ dxhat2));
    @show √mean([mean(abs2, yϵ[:,1] .- corrected_signal_instance(xhat1, dxhat1, ϵhat1)) for _ in 1:1000]);
    @show √mean([mean(abs2, yϵ[:,1] .- corrected_signal_instance(xhat2, dxhat2, ϵhat2)) for _ in 1:1000]);
end;
=#

####
#### Toy problem MCMC inference
####

#=
Turing.@model toy_model_rician_noise(
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
        chain = sample(toy_model_rician_noise(y, correction_and_noiselevel), NUTS(), 1000; verbose = true, init_theta = theta0)
        # chain = psample(toy_model_rician_noise(y, correction_and_noiselevel), NUTS(), 1000, 3; verbose = true, init_theta = theta0)
        callback(y, chain) && return chain
    end
end
function toy_theta_mcmc_inference(Y::AbstractMatrix, args...; kwargs...)
    tasks = map(1:size(Y,2)) do j
        Threads.@spawn signal_loglikelihood_inference(Y[:,j], initial_guess, args...; kwargs...)
    end
    return map(Threads.fetch, tasks)
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
    correction_and_noiselevel = let _model = deepcopy(BSON.load("/home/jon/Documents/UBCMRI/BlochTorreyExperiments-master/MMDLearning/output/2020-02-20T15:43:48.506/best-model.bson")["model"]) #deepcopy(model)
        function(x)
            out = _model(x)
            dx, logϵ = out[1:end÷2], out[end÷2+1:end]
            return abs.(x .+ dx), exp.(logϵ)
        end
    end
    signal_model = function(θhat)
        x = toy_signal_model(θhat, nothing, 4)
        xhat, ϵhat = correction_and_noiselevel(x)
        # zR = ϵhat .* randn(size(x)...)
        # zI = ϵhat .* randn(size(x)...)
        # yhat = @. sqrt((xhat + zR)^2 + zI^2)
        yhat = rand.(Rician.(xhat, ϵhat))
    end
    fitresults = function(y, c)
        θhat = map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst])
        # ϵhat = 10^map(k -> mean(c[k])[1,:mean], [:logeps])[1]
        # yhat, ϵhat = correction_and_noiselevel(toy_signal_model(θhat, nothing, 4))
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

    # @time cs = toy_theta_mcmc_inference(Y, correction_and_noiselevel);
    # res = map(j -> fitresults(Y[:,j], cs[j]), 1:size(Y,2))
    # ps = map(j -> plotresults(Y[:,j], cs[j]), 1:size(Y,2))
    # θhat = reduce(hcat, map(k -> mean(c[k])[1,:mean], [:freq, :phase, :offset, :amp, :tconst]) for c in cs)
    # Yerr = sort(getfield.(res, :yerr))

    @time bbres = signal_loglikelihood_inference(Y, nothing, signal_model);
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
