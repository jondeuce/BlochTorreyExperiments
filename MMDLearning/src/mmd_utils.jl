####
#### Signal model
####

function signal_data(
        image::Array{T,4},
        batchsize = nothing;
        threshold = T(opts.Threshold)
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

function theta_bounds(T = Float64;
        ntheta = settings["data"]["ntheta"]::Int,
    )
    theta_lb = T[120.0, 120.0,    8.0,    8.0, 0.0]
    theta_ub = T[180.0, 180.0, 1000.0, 1000.0, 1.0]
    theta_bd = collect(zip(theta_lb, theta_ub))
    @assert ntheta == length(theta_bd)
    return theta_bd
end
theta_sampler(args...; kwargs...) = broadcast(bound -> bound[1] + (bound[2]-bound[1]) * rand(typeof(bound[1])), theta_bounds(args...; kwargs...))

noise_model!(buffer::AbstractVector{T}, signal::AbstractVector{T}, ϵ::AbstractVector{T}) where {T} = (randn!(buffer); buffer .*= T(ϵ[1] * signal[1]); buffer)
noise_model(signal::AbstractVector, ϵ::AbstractVector) = (ϵ[1] * signal[1]) .* randn(eltype(signal), length(signal))
# noise_model!(buffer::AbstractVector{T}, signal::AbstractVector{T}, ϵ::AbstractVector{T}) where {T} = (randn!(buffer); buffer .*= ϵ .* signal[1]; buffer)
# noise_model(signal::AbstractVector, ϵ::AbstractVector) = ϵ .* signal[1] .* randn(eltype(signal), length(signal))

function signal_model_work(T = Float64; nTE = opts.nTE::Int)
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
        ϵ::AbstractVector;
        TE::T = T(opts.TE)
    ) where {T}
    @unpack epg_work, signal, real_noise, imag_noise = work
    refcon  = θ[1]
    alpha   = θ[2]
    T2short = θ[3]/1000
    dT2     = θ[4]/1000
    Ashort  = θ[5]
    T2long  = T2short + dT2
    Along   = 1-Ashort
    T1short = T(1000.0)/1000
    T1long  = T(1000.0)/1000

    signal  .= zero(T)
    short    = DECAES.EPGdecaycurve!(epg_work, length(signal), alpha, TE, T2short, T1short, refcon)
    signal .+= (Ashort / sum(short)) .* short
    long     = DECAES.EPGdecaycurve!(epg_work, length(signal), alpha, TE, T2long,  T1long,  refcon)
    signal .+= (Along / sum(long)) .* long
    signal ./= sum(signal)

    # Add noise to "real" and "imag" channels in quadrature
    if eltype(ϵ) == eltype(θ)
        if !all(==(0), ϵ)
            noise_model!(real_noise, signal, ϵ) # populate real_noise
            noise_model!(imag_noise, signal, ϵ) # populate imag_noise
            signal .= sqrt.((signal .+ real_noise).^2 .+ imag_noise.^2)
            signal ./= sum(signal)
        end
        return signal
    else
        # Add (forward-differentiable) noise to "real" and "imag" channels in quadrature
        signal = sqrt.((signal .+ noise_model(signal, ϵ)).^2 .+ noise_model(signal, ϵ).^2)
        signal ./= sum(signal)
        return signal
    end
end
signal_model(θ::AbstractVector{T}, ϵ::AbstractVector) where {T} = signal_model!(signal_model_work(T), θ, ϵ)

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
mmd_bandwidth_optfun(logσ::AbstractVector, args...; kwargs...) = mmd_bandwidth_optfun(logσ[], args...; kwargs...)
∇mmd_bandwidth_optfun(logσ::AbstractVector, args...; kwargs...) = [∇mmd_bandwidth_optfun(logσ[], args...; kwargs...)]

function mmd_flux_bandwidth_optfun(logσ::AbstractVector, X, Y)
    m = size(X,2)
    MMDsq  = m * mmd_flux(logσ, X, Y)
    MMDvar = m^2 * mmdvar_flux(logσ, X, Y)
    
    # Avoiding div by zero/negative:
    #   m^2 * MMDvar >= ϵ  -->  m * MMDσ >= √ϵ
    ϵ = eps(typeof(MMDvar))
    t = MMDsq / √max(MMDvar, ϵ)
    return t
end
function ∇mmd_flux_bandwidth_optfun(logσ::AbstractVector, args...; kwargs...)
    if length(logσ) <= 16
        ForwardDiff.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)
    else
        Flux.Zygote.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)[1]
    end
end
∇mmd_flux_bandwidth_optfun_fwddiff(logσ::AbstractVector, args...; kwargs...) = ForwardDiff.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)
∇mmd_flux_bandwidth_optfun_zygote(logσ::AbstractVector, args...; kwargs...) = Flux.Zygote.gradient(logσ -> mmd_flux_bandwidth_optfun(logσ, args...; kwargs...), logσ)[1]

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
#### GMM data samplers from learned prior
####

function make_gmm_data_samplers(
        image;
        input_noise = false,
        ntheta = settings["data"]["ntheta"]::Int,
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

    sampleX = if input_noise
        signal_work = signal_model_work(Float64)
        function(batchsize, noise)
            θ = sampleθ(batchsize)
            X = reduce(hcat, [copy(signal_model!(signal_work, θ[:,j], noise)) for j in 1:batchsize])
            return X
        end
    else
        signal_work = signal_model_work(Float64)
        function(batchsize)
            θ = sampleθ(batchsize)
            X = reduce(hcat, [copy(signal_model!(signal_work, θ[:,j], [0.0])) for j in 1:batchsize])
            return X
        end
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
    y = @. (offset + amp * abs(sin(2pi * freq * ts - phase))^power) * exp(-ts / tconst)
    if !isnothing(epsilon)
        zR = epsilon .* randn(size(y)...)
        zI = epsilon .* randn(size(y)...)
        @. y = sqrt((y + zR)^2 + zI^2)
    end
    return y
end
toy_signal_model(n::Int, args...; kwargs...) = toy_signal_model(toy_theta_sampler(n), args...; kwargs...)

function toy_theta_sampler(n::Int = 1)
    unif(a, b) = a .+ (b-a) .* rand(n)

    # freq   = unif(1/64,  1/16)
    # freq   = unif(1/64,  1/64)
    freq   = unif(1/64,  1/32)
    phase  = unif( 0.0,    pi)
    offset = unif( 0.25,  0.5)
    amp    = unif( 0.1,  0.25)
    tconst = unif(16.0, 128.0)

    return permutedims(hcat(freq, phase, offset, amp, tconst))
end

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

nothing
