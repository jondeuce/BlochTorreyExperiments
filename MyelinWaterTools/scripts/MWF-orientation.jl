# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlobalUtils
using MWFUtils, MWFLearning, DECAES
using SpecialFunctions, Optim, BlackBoxOptim

include(joinpath("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/src/rician.jl"))

# [..] all MWI sequences had 48 echoes with the first echo at TE = 8 ms, ∆TE = 8 ms
# [..] T2 range of 8 ms to 2.0 s
# [..] myelin water component is defined as the T2 times of the distribution between 8 ms and 25 ms
# [..] intra- and extra-cellular water component is defined as the T2 times above 25 ms
# [..] cut-off between the two water pools was set to 25 ms, which was based on the measured T2 distributions

const default_t2mapopts = T2mapOptions{Float64}(
    MatrixSize = (1, 1, 1),
    nTE = 48,
    nT2 = 40,
    TE = 8e-3,
    T2Range = (8e-3, 2.0),
    Threshold = 0.0,
    Silent = false,
    SaveNNLSBasis = false,
)

const default_t2partopts = T2partOptions{Float64}(
    MatrixSize = default_t2mapopts.MatrixSize,
    nT2 = default_t2mapopts.nT2,
    T2Range = default_t2mapopts.T2Range,
    SPWin = (prevfloat(default_t2mapopts.T2Range[1]), 25e-3), # ensure T2Range[1] is captured
    MPWin = (nextfloat(25e-3), nextfloat(default_t2mapopts.T2Range[2])), # ensure disjoint with SPWin + T2Range[2] is captured
    Silent = true,
)

# g=sqrt(1/2), η = 0.8  --> MVF = 0.4,  MWF = 0.25
# g=sqrt(3/4), η = 8/11 --> MVF = 2/11, MWF = 0.10
const default_btp = BlochTorreyParameters{Float64}(
    g_ratio = sqrt(3/4),
    AxonPDensity = 8/11,
    R2_sp = 1/15e-3,
    R2_lp = 1/75e-3,
);

####
#### MWF orientation plots
####

function fractions(p::BlochTorreyParameters)
    @unpack PD_sp, PD_lp, g_ratio, MWF, MVF, AxonPDensity = p

    EVF = 1 - AxonPDensity # external tissue area = 1 - axon area (includes myelin + intra-celluar)
    IVF = g_ratio^2 * AxonPDensity # intracellular area = g^2 * axon area by definition of g
    @assert 1 ≈ MVF + EVF + IVF

    EWF = PD_lp * EVF / (PD_lp * IVF + PD_lp * EVF + PD_sp * MVF)
    IWF = PD_lp * IVF / (PD_lp * IVF + PD_lp * EVF + PD_sp * MVF)
    @assert MWF ≈ PD_sp * MVF / (PD_lp * IVF + PD_lp * EVF + PD_sp * MVF)
    @assert 1 ≈ MWF + EWF + IWF

    return @ntuple(MVF, EVF, IVF, MWF, EWF, IWF)
end

function mwf_vs_orientation(
        p::BlochTorreyParameters{Float64},
        θs::AbstractVector{Float64};
        flip_angle = 180.0,
        SNR = 50.0,
        nsamples,
    )

    @unpack R2_sp, R2_lp, R2_lp_DipDip, R2_Tissue, R1_Tissue = p
    @unpack MVF, EVF, IVF, MWF, EWF, IWF = fractions(p)

    t2mapopts = T2mapOptions(default_t2mapopts; MatrixSize = (nsamples, length(θs), 1))
    t2partopts = T2partOptions(default_t2partopts; MatrixSize = (nsamples, length(θs), 1))

    S = mapreduce(hcat, θs) do θ
        R2_lp_θ = R2_lp + R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2
        # R2_lp_θ = R2_lp
        # R2_sp_θ = R2_sp + R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2
        R2_sp_θ = R2_sp
        S = EWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_Tissue), inv(R1_Tissue), 180.0) .+
            IWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_lp_θ), inv(R1_Tissue), 180.0) .+
            MWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_sp_θ), inv(R1_Tissue), 180.0)
    end # S is [nTE x nθ]

    image = MWFLearning.add_rician(repeat(S, 1, 1, nsamples, 1), SNR) # image is [nTE x nθ x nsamples x 1]
    image = permutedims(image, (3,2,4,1)) # [nsamples x nθ x 1 x nTE]

    maps, dist = T2mapSEcorr(image, t2mapopts)
    parts = T2partSEcorr(dist, t2partopts)

    return @ntuple(image, maps, dist, parts)
end

dT2_lp = 10.0e-3;
T2_lp = inv(default_btp.R2_lp);
R2_lp_DipDip = (inv(T2_lp - dT2_lp) - inv(T2_lp)) / 4;
btp = BlochTorreyParameters(default_btp; R2_lp_DipDip = R2_lp_DipDip);
θs = range(0.0, 90.0; length = 37);

res = mwf_vs_orientation(btp, θs;
    flip_angle = 170.0,
    SNR = 50.0,
    nsamples = 1_000,
);

apply_over_θ(f, x::Array{<:Real,3}) = map(xi -> f(filter(!isnan, vec(xi))), eachslice(x; dims=2));
std_mean(x) = std(x) / sqrt(length(x)); # std of the mean
std_med(x) = eltype(x)(1.253) * std(x) / sqrt(length(x)); # std of the mean
iewf,  σ_iewf  = apply_over_θ.((mean, std_mean), Ref(res.parts["mfr"]));
mwf,   σ_mwf   = apply_over_θ.((mean, std_mean), Ref(res.parts["sfr"]));
t2mw,  σ_t2mw  = apply_over_θ.((mean, std_mean), Ref(1000 .* res.parts["sgm"])) # convert to ms;
t2iew, σ_t2iew = apply_over_θ.((mean, std_mean), Ref(1000 .* res.parts["mgm"])) # convert to ms;
fa,    σ_fa    = apply_over_θ.((mean, std_mean), Ref(res.maps["alpha"]));

@unpack MVF, EVF, IVF, MWF, EWF, IWF = fractions(btp);
t2iew_true = map(θs) do θ
    T2_ewf = inv(btp.R2_Tissue)
    T2_iwf = inv(btp.R2_lp + btp.R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2)
    T2_av = exp((EWF * log(T2_ewf) + IWF * log(T2_iwf)) / (EWF + IWF))
    1000 * T2_av
end;

pyplot(size=(800,600))
pmwf = plot(
    plot(θs, mwf;   ribbon = σ_mwf,   xlab = "theta [deg]", ylab = "mwf [a.u.]",  lab = "mwf", yticks = 0:0.002:1.0),
    plot(θs, t2mw;  ribbon = σ_t2mw,  xlab = "theta [deg]", ylab = "T2 mw [ms]",  lab = "T2 mw"),
    plot() |>
        #p-> plot!(p, θs, t2iew_true; xlab = "theta [deg]", ylab = "T2 iew [ms]", lab = "True") |>
        p -> plot!(p, θs, t2iew;      xlab = "theta [deg]", ylab = "T2 iew [ms]", lab = "T2 iew", ribbon = σ_t2iew),
    plot(θs, iewf;  ribbon = σ_iewf,  xlab = "theta [deg]", ylab = "iewf [a.u.]", lab = "iewf", yticks = 0:0.002:1.0),
    plot(θs, mwf ./ iewf;             xlab = "theta [deg]", ylab = "mwf / iewf",  lab = "mwf / iewf", yticks = 0:0.002:1.0),
    # plot(θs, fa;    ribbon = σ_fa,    xlab = "theta [deg]", ylab = "FA [deg]",   lab = "FA"),
    plot(
        1000 .* default_t2mapopts.TE .* (1:default_t2mapopts.nTE),
        permutedims(res.image[1,[1,end],1,:]); xlab = "Time [ms]", ylab = "Signal", label = "signal (θ = " .* string.(θs[[1 end]]) .* ")",
    ),
    layout = (2,3),
);
display(pmwf);
savefig.(Ref(pmwf), "mwf_vs_orientation" .* [".png", ".pdf"]);

pyplot(size=(800,600))
phist = plot(
    histogram(        vec(res.parts["sfr"]);  xlab = "mwf",        lab = "mwf",    nbins = 50),
    histogram(1000 .* vec(res.parts["sgm"]);  xlab = "t2mw [ms]",  lab = "T2 mw",  nbins = 50),
    histogram(1000 .* vec(res.parts["mgm"]);  xlab = "t2iew [ms]", lab = "T2 iew", nbins = 50),
    histogram(        vec(res.maps["alpha"]); xlab = "FA [deg]",   lab = "FA",     nbins = 50),
    plot(res.image[1,1,1,:]; label = "signal (θ = $(θs[1]))"),
);
display(phist);

####
#### MWF MLE opt
####

const mle_timer = TimerOutput()

const nloptalgs = [
    :LD_LBFGS,
    #:LD_MMA,
    :LD_TNEWTON_PRECOND_RESTART,
    :LD_TNEWTON_PRECOND,
    :LD_TNEWTON_RESTART,
    :LD_TNEWTON,
    :LD_VAR2,
    :LD_VAR1,
]

const opt_failures = Dict{Symbol,Int}(
    :Optim => 0,
    [alg => 0 for alg in nloptalgs]...,
)

function make_rician_mle_loss(
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        λ::Union{T,Nothing} = nothing,
    ) where {T}

    nTE, nT2 = size(A)
    x, ∇x = zeros(T, nT2), zeros(T, nT2)
    ν, ∇ν = zeros(T, nTE), zeros(T, nTE)

    function fg!(F, G, θ::AbstractVector{T_}) where {T_}
        @assert T_ === T # Note: same T as input A, b

        @timeit mle_timer "common" begin
            @inbounds for i in 1:nT2
                x[i] = θ[i] # assign amplitudes
                # x[i] = exp(θ[i]) # assign amplitudes #TODO x -> logx
            end
            # @inbounds σ = θ[end] #TODO sigma -> logsigma
            @inbounds σ = exp(θ[end]) #TODO sigma -> logsigma
            @inbounds mul!(ν, A, x) # assign signal basis, given amplitudes
        end

        if !isnothing(G)
            @timeit mle_timer "gradient" begin
                ∇σ = zero(T_)
                @inbounds for i in 1:nTE
                    ∇ν_, ∇σ_ = ∇logpdf(Rician(ν[i], σ; check_args = false), b[i])
                    ∇ν[i] = -∇ν_ # (negative) logL gradient w.r.t. ν[i]
                    ∇σ -= ∇σ_ # accumulate (negative) logL gradient w.r.t. σ
                end
                @inbounds mul!(∇x, A', ∇ν) # transform ∂(-logL)/∂ν -> ∂(-logL)/∂x

                @inbounds for i in 1:nT2
                    if isnothing(λ)
                        G[i] = ∇x[i] # assign ∂(-logL)/∂x
                    else
                        G[i] = ∇x[i] + λ * x[i] # assign ∂(-logL)/∂x
                    end
                    # if isnothing(λ)
                    #     ∇logx = x[i] * ∇x[i] # transform ∂(-logL)/∂x -> ∂(-logL)/∂(logx) #TODO x -> logx
                    #     G[i] = ∇logx # assign ∂(-logL)/∂(logx) #TODO x -> logx
                    # else
                    #     x_, ∇x_ = x[i], ∇x[i]
                    #     ∇logx = x_ * (∇x_ + λ * x_)
                    #     G[i] = ∇logx # assign ∂(-logL)/∂(logx) #TODO x -> logx
                    # end
                end
                ∇logσ = σ * ∇σ # transform ∂(-logL)/∂σ -> ∂(-logL)/∂(logσ)
                @inbounds G[end] = ∇logσ # assign ∂(-logL)/∂(logσ)
            end
        end

        if !isnothing(F)
            @timeit mle_timer "function" begin
                ℓ = zero(T_)
                @inbounds for i in 1:nTE
                    ℓ -= logpdf(Rician(ν[i], σ; check_args = false), b[i])
                end
                if !isnothing(λ)
                    @inbounds for i in 1:nT2
                        ℓ += λ/2 * x[i]^2
                    end
                end
                return ℓ
            end
        end

        return nothing
    end
end

function loglikelihood_inference(
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64},
        initial_guess::Union{AbstractVector{Float64},Nothing} = nothing,
        λ::Union{Float64,Nothing} = nothing;
        verbose = false,
        bbopt_kwargs = Dict(:MaxTime => 5.0, :TraceMode => :Verbose),
    )
    # Deterministic loss function, suitable for Optim
    nTE, nT2 = size(A)
    mle_loss! = make_rician_mle_loss(A, b, λ)
    SearchRange = vcat(
        fill((0.0, 1.0), nT2), # T2 components
        [(-10.0, -1.0)], # ln(noise level)
    )

    bbres = nothing
    if isnothing(initial_guess)
        bbres = bboptimize(θ -> mle_loss!(true, nothing, θ);
            SearchRange = SearchRange,
            TraceMode = :silent,
            bbopt_kwargs...
        )
    end

    lo, hi = (x->x[1]).(SearchRange), (x->x[2]).(SearchRange)
    θ0 = isnothing(initial_guess) ? BlackBoxOptim.best_candidate(bbres) : initial_guess
    
    @timeit mle_timer "Optim" begin
        inner_alg_type = Optim.LBFGS
        inner_alg = inner_alg_type(m = 10) #Optim.ConjugateGradient(), Optim.GradientDescent()
        optres = Optim.optimize(Optim.only_fg!(mle_loss!), lo, hi, θ0, Optim.Fminbox(inner_alg))
        # optres = Optim.optimize(Optim.only_fg!(mle_loss!), θ0, Optim.LBFGS(m = 10)) #TODO x -> logx
        verbose && println("Optim $(inner_alg_type):\n\tgot $(Optim.minimum(optres)) after $(Optim.iterations(optres)) iterations")
    end

    setsigma!(x) = (x[end] = log(std(b)/nT2); x)
    nloptres = Dict{Symbol,Any}(
        map(nloptalgs) do alg
            alg => Dict{Symbol,Any}(
                :opt  => NLopt.Opt(alg, nT2+1),
                # :x    => rand(nT2+1) |> setsigma!,
                # :x    => fill(maximum(b)/nT2, nT2+1) |> setsigma!,
                # :x    => zeros(nT2+1) |> setsigma!, #doesn't work
                :x    => copy(θ0),
                :minf => nothing,
                :minx => nothing,
                :ret  => nothing,
            )
        end
    )

    for alg in nloptalgs
        nloptres[alg][:opt].lower_bounds = [zeros(nT2); -Inf]
        nloptres[alg][:opt].xtol_rel = 1e-4
        # nloptres[alg][:opt].ftol_rel = 1e-4
        nloptres[alg][:opt].min_objective = function f(x::Vector, grad::Vector)
            if length(grad) > 0
                mle_loss!(true, grad, x)
            else
                mle_loss!(true, nothing, x)
            end
        end

        @timeit mle_timer "NLopt ($alg)" begin
            nloptres[alg][:minf], nloptres[alg][:minx], nloptres[alg][:ret] = NLopt.optimize!(nloptres[alg][:opt], nloptres[alg][:x])

            verbose && println("NLopt $(nloptres[alg][:opt].algorithm):\n\tgot $(nloptres[alg][:minf]) after $(nloptres[alg][:opt].numevals) iterations (returned $(nloptres[alg][:ret]))")
        end
    end

    bestf = min(Optim.minimum(optres), minimum([nloptres[alg][:minf] for alg in nloptalgs]))
    if Optim.minimum(optres) > bestf + 1
        opt_failures[:Optim] += 1
    end
    foreach(nloptalgs) do alg
        if nloptres[alg][:minf] > bestf + 1
            opt_failures[alg] += 1
        end
    end

    # θ = best_candidate(bbres)
    θ = Optim.minimizer(optres)
    # θ = nloptres[:minx]
    x = θ[1:end-1]
    # x = exp.(θ[1:end-1]) #TODO x -> logx
    ν = A * x
    σ = exp(θ[end])

    return @ntuple(θ, x, ν, σ)
end

foreach(k -> opt_failures[k] = 0, keys(opt_failures))
TimerOutputs.reset_timer!(mle_timer)

# let I = CartesianIndex(3,1,1)
for I in CartesianIndices((5, size(res.image,2), 1))
    @unpack nT2, nTE, T2Range = default_t2mapopts
    T2_times = DECAES.logrange(T2Range..., nT2)
    nnls_t2dist = res.dist[I,:]
    noisy_signal = res.image[I,:]
    # decay_basis = res.maps["decaybasis"][I,:,:]
    decay_basis = zeros(nTE, nT2)
    lambda = 1.0
    DECAES.epg_decay_basis!(DECAES.EPGdecaycurve_work(Float64, nTE), decay_basis, 170.0, T2_times, default_t2mapopts)

    σ0 = std(noisy_signal - decay_basis * nnls_t2dist)
    θ0 = [max.(nnls_t2dist, 1e-3*maximum(nnls_t2dist)); log(σ0)] # init with NNLS solution
    # θ0 = [fill(noisy_signal[1]/nT2, nT2); -5.0]
    # θ0 = [zeros(nT2); log(1e-3)] #TODO x -> logx

    # foreach(k -> opt_failures[k] = 0, keys(opt_failures))
    out = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda; verbose = false);
    display(opt_failures)

    # TimerOutputs.reset_timer!(mle_timer)
    for i in 1:100
        out = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda);
    end
    TimerOutputs.print_timer(mle_timer)
    println("")

    t2dist = out.x
    sigma = out.σ
    signal = out.ν
    SNR = -20 * log10(sigma / signal[1])
    plot(T2_times, [t2dist nnls_t2dist]; lab = ["mle" "nnls"], xscale = :log10, xformatter = x->string(round(x;sigdigits=3))) |> display
    @show σ0, sigma, SNR
    display(T2partSEcorr(reshape(t2dist,1,1,1,:), default_t2partopts))
end;

#= make_rician_mle_loss
let
    A = rand(300,150)
    b = rand(300)
    λ = rand() #nothing
    fg! = make_rician_mle_loss(A,b,λ)
    
    nTE, nT2 = size(A)
    θ, ∇θ = rand(nT2+1), zeros(nT2+1)
    θ[end] = -abs(θ[end]) #logσ

    # Finite diff gradient test
    δ = cbrt(eps())
    ∇θδ = zeros(nT2+1)
    for i in 1:length(θ)
        tmp = θ[i]
        θ[i] = tmp - δ/2
        y1 = fg!(true, nothing, θ)
        θ[i] = tmp + δ/2
        y2 = fg!(true, nothing, θ)
        θ[i] = tmp
        ∇θδ[i] = (y2-y1)/δ
    end
    fg!(nothing, ∇θ, θ) # compute exact gradient
    display(maximum(abs, (∇θ .- ∇θδ) ./ ∇θ))

    # Benchmarking
    # @btime $(fg!)(true, nothing, $θ)
    # @btime $(fg!)(nothing, $∇θ, $θ)
    # @btime $(fg!)(true, $∇θ, $θ)
end;
=#
