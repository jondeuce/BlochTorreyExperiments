# Activate project and load packages for this script
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GlobalUtils
using MWFUtils, MWFLearning, DECAES
using SpecialFunctions, Optim, BlackBoxOptim, NLopt, Roots
pyplot(size=(800,600))

include(joinpath(@__DIR__, "../../MMDLearning/src/rician.jl"))

# [..] all MWI sequences had 48 echoes with the first echo at TE = 8 ms, ∆TE = 8 ms
# [..] T2 range of 8 ms to 2.0 s
# [..] myelin water component is defined as the T2 times of the distribution between 8 ms and 25 ms
# [..] intra- and extra-cellular water component is defined as the T2 times above 25 ms
# [..] cut-off between the two water pools was set to 25 ms, which was based on the measured T2 distributions

const default_t2mapopts = T2mapOptions{Float64}(
    legacy = true,
    MatrixSize = (1, 1, 1),
    nTE = 48,
    nT2 = 40,
    TE = 8e-3,
    T2Range = (8e-3, 2.0),
    Threshold = 0.0,
    Silent = true,
    SaveNNLSBasis = false,
    SaveRegParam = true,
)

const default_t2partopts = T2partOptions{Float64}(
    legacy = true,
    MatrixSize = default_t2mapopts.MatrixSize,
    nT2 = default_t2mapopts.nT2,
    T2Range = default_t2mapopts.T2Range,
    SPWin = (prevfloat(default_t2mapopts.T2Range[1]), 25e-3), # ensure T2Range[1] is captured
    MPWin = (nextfloat(25e-3), nextfloat(default_t2mapopts.T2Range[2])), # ensure disjoint with SPWin and T2Range[2] is captured
    Silent = true,
)

# g = sqrt(1/2), η = 0.8          --> MWF = 0.25, MVF = 0.4
# g = 0.7920411558414019, η = 0.7 --> MWF = 0.15, MVF = 0.2608695652173913
# g = 0.8329931278350429, η = 0.7 --> MWF = 0.12, MVF = 0.2142857142857143
# g = sqrt(3/4), η = 8/11         --> MWF = 0.10, MVF = 2/11
const default_btp = BlochTorreyParameters{Float64}(
    g_ratio = sqrt(3/4),
    AxonPDensity = 8/11,
    R2_sp = 1/15e-3,
    R2_lp = 1/75e-3,
);
# @show(fractions(default_btp));

####
#### MWF orientation plots
####

#=
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
        # R2_lp_θ = R2_lp + R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2
        # R2_sp_θ = R2_sp
        R2_lp_θ = R2_lp #TODO
        R2_sp_θ = R2_sp + R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2 #TODO
        # @show θ, 1000/R2_sp_θ
        S = EWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_Tissue), inv(R1_Tissue), 180.0) .+
            IWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_lp_θ), inv(R1_Tissue), 180.0) .+
            MWF .* EPGdecaycurve(t2mapopts.nTE, flip_angle, t2mapopts.TE, inv(R2_sp_θ), inv(R1_Tissue), 180.0)
    end # S is [nTE x nθ]

    image = MWFLearning.add_rician(repeat(S, 1, 1, nsamples, 1), SNR) # image is [nTE x nθ x nsamples x 1]
    image = permutedims(image, (3,2,4,1)) # [nsamples x nθ x 1 x nTE]

    maps, dist = T2mapSEcorr(image, t2mapopts)
    parts = T2partSEcorr(dist, t2partopts)

    return @dict(image, maps, dist, parts)
end

# for default_SNR in 30.0:10.0:100.0
default_SNR = 40.0
default_flip_angle = 165.0

# dT2 = 10.0e-3;
# R2_lp_DipDip = (inv(inv(default_btp.R2_lp) - dT2) - default_btp.R2_lp) / 4;
dT2 = 0.0e-3; #TODO for R2_sp
R2_lp_DipDip = (inv(inv(default_btp.R2_sp) - dT2) - default_btp.R2_sp) / 4; #TODO for R2_sp
btp = BlochTorreyParameters(default_btp; R2_lp_DipDip = R2_lp_DipDip);
θs = range(0.0, 90.0; length = 37);

@time res = mwf_vs_orientation(btp, θs;
    flip_angle = default_flip_angle,
    SNR = default_SNR,
    nsamples = 1_000,
);

res[:mle_dist] = similar(res[:dist]);
@time DECAES.tforeach(CartesianIndices(size(res[:image])[1:3])) do I
    nnls_t2dist = res[:dist][I,:]
    noisy_signal = res[:image][I,:]
    decay_basis = res[:maps]["decaybasis"][I,:,:]
    σ0 = std(noisy_signal - decay_basis * nnls_t2dist)
    μ = res[:maps]["mu"][I]
    # lambda = nothing
    lambda = 0.01 * 2*μ^2/σ0^2
    # lambda = 2*μ^2/σ0^2
    θ0 = [max.(nnls_t2dist, 1e-3*maximum(nnls_t2dist)); log(σ0)] # init with NNLS solution
    @unpack θ, x = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda, nlopt_solver; verbose = false);
    res[:mle_dist][I,:] .= x
end;
res[:mle_parts] = T2partSEcorr(res[:mle_dist], T2partOptions(default_t2partopts; MatrixSize = size(res[:image])[1:3]));

for (dist, parts) in [(:dist,:parts), (:mle_dist,:mle_parts)]
    apply_over_θ(f, x::Array{<:Real,3}) = map(xi -> f(filter(!isnan, vec(xi))), eachslice(x; dims=2));
    std_mean(x) = std(x) / sqrt(length(x)); # std of the mean
    std_med(x) = eltype(x)(1.253) * std(x) / sqrt(length(x)); # std of the mean
    iewf,  σ_iewf  = apply_over_θ.((mean, std_mean), Ref(res[parts]["mfr"]));
    mwf,   σ_mwf   = apply_over_θ.((mean, std_mean), Ref(res[parts]["sfr"]));
    t2mw,  σ_t2mw  = apply_over_θ.((mean, std_mean), Ref(1000 .* res[parts]["sgm"])) # convert to ms;
    t2iew, σ_t2iew = apply_over_θ.((mean, std_mean), Ref(1000 .* res[parts]["mgm"])) # convert to ms;
    fa,    σ_fa    = apply_over_θ.((mean, std_mean), Ref(res[:maps]["alpha"]));

    @unpack MVF, EVF, IVF, MWF, EWF, IWF = fractions(btp);
    # t2iew_true = map(θs) do θ
    #     T2_ewf = inv(btp.R2_Tissue)
    #     T2_iwf = inv(btp.R2_lp + btp.R2_lp_DipDip * (3 * cosd(θ)^2 - 1)^2)
    #     T2_av = exp((EWF * log(T2_ewf) + IWF * log(T2_iwf)) / (EWF + IWF))
    #     1000 * T2_av
    # end;

    pyplot(size=(800,600))
    s = x -> string(x ≈ round(Int, x) ? round(Int, x) : round(x; sigdigits = 3))
    titlestr = "SNR = $(s(default_SNR)), MWF = $(s(MWF)), $(s(1000/default_btp.R2_sp-1000*dT2)) ms ≤ T2 mw ≤ $(s(1000/default_btp.R2_sp)) ms, T2 iew = $(s(1000/default_btp.R2_lp)) ms, flip = $(s(default_flip_angle)) deg"

    pmwf = plot(
        plot(title = titlestr, titlefontsize = 10, grid = false, showaxis = false),
        plot(
            plot(θs, mwf;   ribbon = σ_mwf,  xlab = "theta [deg]", ylab = "mwf [a.u.]",       lab = "mwf"),#, yticks = 0:0.002:1.0),
            plot(θs, t2mw;  ribbon = σ_t2mw, xlab = "theta [deg]", ylab = "T2 mw [ms]",       lab = "T2 mw"),
            plot() |>
                #p-> plot!(p, θs, t2iew_true; xlab = "theta [deg]", ylab = "T2 iew [ms]", lab = "True") |>
                p -> plot!(p, θs, t2iew;      xlab = "theta [deg]", ylab = "T2 iew [ms]", lab = "T2 iew", ribbon = σ_t2iew),
            plot(θs, iewf;  ribbon = σ_iewf, xlab = "theta [deg]", ylab = "iewf [a.u.]",      lab = "iewf"),#, yticks = 0:0.002:1.0),
            plot(θs, fa;    ribbon = σ_fa,   xlab = "theta [deg]", ylab = "flip angle [deg]", lab = "flip angle"),
            plot(
                1000 .* default_t2mapopts.TE .* (1:default_t2mapopts.nTE),
                permutedims(res[:image][1,[1,end],1,:]); xlab = "time [ms]", ylab = "signal", label = "signal (θ = " .* string.(θs[[1 end]]) .* ")",
            ),
            layout = (2,3),
        );
        layout = @layout([a{0.01h}; b]),
    );
    display(pmwf);
    savefig.(Ref(pmwf), "output/mwf_vs_orientation_SNR=$default_SNR" .* [".png", ".pdf"]);

    phist = plot(
        plot(title = titlestr, titlefontsize = 10, grid = false, showaxis = false),
        plot(
            histogram(        vec(res[parts]["sfr"]);  xlab = "mwf",              lab = "mwf",        nbins = 50),
            histogram(1000 .* vec(res[parts]["sgm"]);  xlab = "t2mw [ms]",        lab = "T2 mw",      nbins = 50),
            histogram(1000 .* vec(res[parts]["mgm"]);  xlab = "t2iew [ms]",       lab = "T2 iew",     nbins = 50),
            histogram(        vec(res[:maps]["alpha"]); xlab = "flip angle [deg]", lab = "flip angle", nbins = 50),
            plot(res[:image][1,1,1,:]; label = "signal (θ = $(θs[1]))"),
        );
        layout = @layout([a{0.01h}; b]),
    );
    display(phist);
    savefig.(Ref(phist), "output/mwf_hist_SNR=$default_SNR" .* [".png", ".pdf"]);
end # for (dist, parts) in ...

# error("mwf done")
=#

####
#### MWF orientation of BT simulations
####

using BSON
pyplot(size=(1600,900))

#=
const sim_dirs = [
    "/project/st-arausch-1/jcd1994/MWI-Orientation/mwi-orient-1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-10X-chi-1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-50X-chi-1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-100X-chi-1",
    # "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-1000X-chi-1",
]
=#
const sim_dirs = [
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=1.0X-Rmu=0.5-v1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=1.0X-Rmu=1.0-v1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=1.0X-Rmu=1.5-v1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=10.0X-Rmu=0.5-v1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=10.0X-Rmu=1.0-v1",
    "/project/st-arausch-1/jcd1994/simulations/ismrm2020/mwi-orient-chi=10.0X-Rmu=1.5-v1",
]

ps = map(sim_dirs) do sim_dir
    meas_dir = joinpath(sim_dir, "measurables")
    bt_sims = map(readdir(meas_dir)[1:min(end,1000)]) do file #TODO [1:min(end,500)]
        BSON.load(joinpath(meas_dir, file))
    end;
    bt_filtered = filter(bt_sims) do bt
        true #175.0 <= rad2deg(bt[:solverparams_dict][:flipangle])
    end;
    bt_filtered = bt_filtered

    mwf = (bt -> bt[:geomparams_dict][:mwf]).(bt_filtered);
    alpha = (bt -> rad2deg(bt[:solverparams_dict][:flipangle])).(bt_filtered);
    theta = (bt -> rad2deg(bt[:btparams_dict][:theta])).(bt_filtered);

    signals = mapreduce(hcat, bt_filtered) do bt
        S = norm.(transverse.(bt[:signals][1 .+ 20 .* (1:default_t2mapopts.nTE)]))
        return S ./ sum(S)
    end;
    maps, dist, parts = let
        image = permutedims(reshape(signals, size(signals)..., 1, 1), (2,4,3,1))
        image = MWFLearning.add_rician(image, 50.0)
        @show size(image)
        t2mapopts = T2mapOptions(default_t2mapopts; MatrixSize = size(image)[1:3])
        t2partopts = T2partOptions(default_t2partopts; MatrixSize = size(image)[1:3])
        maps, dist = T2mapSEcorr(image, t2mapopts)
        parts = T2partSEcorr(dist, t2partopts)
        maps, dist, parts
    end;

    function makebinned(x,y,edges)
        nbins = length(edges)-1
        xmeans, ymeans, ystds, ymeanstds = [zeros(nbins) for _ in 1:4]
        for i in 1:nbins
            f = i < nbins ?
                z -> edges[i] ≤ z < edges[i+1] :
                z -> edges[i] ≤ z ≤ edges[i+1]
            idx = findall(f, x)
            xmeans[i] = mean(edges[i:i+1]) #mean(x[idx])
            ymeans[i] = mean(y[idx])
            ystds[i] = std(y[idx]) # standard error
            ymeanstds[i] = std(y[idx]) / sqrt(length(idx)) # standard error of the mean
        end
        return @ntuple(xmeans, ymeans, ystds, ymeanstds)
    end

    plot(
        let # mwf vs. theta
            x, y, σ, σμ = makebinned(theta, vec(parts["sfr"]), 0:5:90)
            plot(x, [y y]; ribbon = [σ σμ], xlab = "Theta [deg]", ylab = "MWF [a.u.]",
                xlim = (0,90), xticks = 0:15:90, #ylim = (0,0.15),
                marker = (0,), line = (:black,1), leg = :none) #marker = (3,:circle,:black)
        end,
        let # T2 mw vs. theta
            x, y, σ, σμ = makebinned(theta, 1000 .* vec(parts["sgm"]), 0:5:90)
            plot(x, [y y]; ribbon = [σ σμ], xlab = "Theta [deg]", ylab = "T2 mw [ms]",
                xlim = (0,90), xticks = 0:15:90, #ylim = (0,15),
                marker = (0,), line = (:black,1), leg = :none) #marker = (3,:circle,:black)
        end,
        let # T2 iew vs. theta
            x, y, σ, σμ = makebinned(theta, 1000 .* vec(parts["mgm"]), 0:5:90)
            plot(x, [y y]; ribbon = [σ σμ], xlab = "Theta [deg]", ylab = "T2 iew [ms]",
                xlim = (0,90), xticks = 0:15:90, #ylim = (50,80),
                marker = (0,), line = (:black,1), leg = :none) #marker = (3,:circle,:black)
        end,
        let
            xfrmt = x -> round(x; digits=1) |> string
            x = 1000 .* maps["t2times"]
            yall = permutedims(dist[:,1,1,:])
            ymid = mean(yall, dims=2)
            # ymid = median(yall; dims = 2)
            yscale = maximum(ymid)
            yallscale = sqrt.(yall ./ yscale)
            ymu, ysig = mean(yallscale; dims = 2), 3 .* std(yallscale; dims = 2)
            ylo, yhi = min.(ysig, ymu), ysig
            # plot(x, yall; line = (1,:blue,0.05), xlab = "T2 [ms]", ylab = "Amplitude [a.u.]", xscale = :log10, xticks = x[1:3:end], xrot = 45, xformatter = xfrmt, leg = :none)
            plot(x, ymu; line = (1,:blue), ribbon = (ylo, yhi), xlab = "T2 [ms]", ylab = "√(Mean Amplitude) [a.u.]", xscale = :log10, xticks = x[1:3:end], xrot = 45, xformatter = xfrmt, leg = :none)
        end;
        xguidefontsize = 8, yguidefontsize = 8, xtickfontsize = 8, ytickfontsize = 8,
    );
end

for (p, sim_dir) in zip(ps, sim_dirs)
    # display(p)
    # map(suf -> savefig(p, "$(Dates.now())-$(basename(sim_dir)).$suf"), ["png", "pdf"]);
    # map(suf -> savefig(p, "$(basename(sim_dir)).$suf"), ["png", "pdf"]);
end
#=
=#

porient = plot(
    mapreduce(vcat, enumerate(ps)) do (i,p)
        chifact = parse(Float64, first(match(r"chi=(\d+.\d+)X-", sim_dirs[i]).captures))
        Rmu = parse(Float64, first(match(r"Rmu=(\d+.\d+)-", sim_dirs[i]).captures))
        plot(
            plot(title = "χ myelin factor = $(chifact)X, mean radius = $(Rmu) μm", grid = false, showaxis = false, titlefontsize = 10),
            p;
            layout = @layout([a{0.01h}; b]),
        )
    end...,
);
display(porient);
# map(suf -> savefig(porient, "mwi-orient-norm.$suf"), ["png", "pdf"]);

error("plotted")

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
const nloptalg = Ref(:LD_LBFGS)

const optres = Dict{Symbol,Any}(
    :Optim => nothing,
    [alg => nothing for alg in nloptalgs]...,
)

const opt_failures = Dict{Symbol,Int}(
    :Optim => 0,
    [alg => 0 for alg in nloptalgs]...,
)

function make_rician_mle_loss(
        A::AbstractMatrix{T},
        b::AbstractVector{T},
        λ::Union{T,Nothing} = nothing,
        sigma::Union{T,Nothing} = nothing,
    ) where {T}

    nTE, nT2 = size(A)
    x, ∇x = zeros(T, nT2), zeros(T, nT2)
    ν, ∇ν = zeros(T, nTE), zeros(T, nTE)

    function fg!(F, G, θ::AbstractVector{T_}) where {T_}
        @assert T_ === T # Note: same T as input A, b

        @timeit mle_timer "common" begin
            @inbounds @simd for i in 1:nT2
                x[i] = θ[i] # assign amplitudes
                # x[i] = exp(θ[i]) # assign amplitudes #TODO x -> logx
            end
            σ = if isnothing(sigma)
                # @inbounds σ = θ[end] #TODO sigma -> logsigma
                @inbounds σ = exp(θ[end]) #TODO sigma -> logsigma
            else
                sigma
            end
            @inbounds mul!(ν, A, x) # assign signal basis, given amplitudes
        end

        if !isnothing(G)
            @timeit mle_timer "gradient" begin
                ∇σ = zero(T_)
                @inbounds @simd for i in 1:nTE
                    if isnothing(sigma)
                        ∇ν_, ∇σ_ = ∇logpdf(Rician{T_}(ν[i], σ), b[i])
                        ∇ν[i] = -∇ν_ # (negative) logL gradient w.r.t. ν[i]
                        ∇σ -= ∇σ_ # accumulate (negative) logL gradient w.r.t. σ
                    else
                        ∇ν[i] = -∂logpdf_∂ν(Rician{T_}(ν[i], σ), b[i]) # (negative) logL gradient w.r.t. ν[i]
                    end
                end
                @inbounds mul!(∇x, A', ∇ν) # transform ∂(-logL)/∂ν -> ∂(-logL)/∂x

                @inbounds @simd for i in 1:nT2
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
                if isnothing(sigma)
                    ∇logσ = σ * ∇σ # transform ∂(-logL)/∂σ -> ∂(-logL)/∂(logσ)
                    @inbounds G[end] = ∇logσ # assign ∂(-logL)/∂(logσ)
                end
            end
        end

        if !isnothing(F)
            @timeit mle_timer "function" begin
                ℓ = zero(T_)
                @inbounds @simd for i in 1:nTE
                    ℓ -= logpdf(Rician{T_}(ν[i], σ), b[i])
                end
                if !isnothing(λ)
                    @inbounds @simd for i in 1:nT2
                        ℓ += λ/2 * x[i]^2
                    end
                end
                return ℓ
            end
        end

        return nothing
    end
end

function bbopt_solver(fg!, θ0; verbose = false, fixedsigma = false, precond = nothing)
    res = bboptimize(θ -> fg!(true, nothing, θ);
        SearchRange = fixedsigma ?
            fill((0.0, 1.0), length(θ0)) :
            push!(fill((0.0, 1.0), length(θ0)-1), (-10.0, -1.0)),
        TraceMode = verbose ? :Verbose : :Silent,
        MaxTime = 5.0,
    )
    θ = BlackBoxOptim.best_candidate(res)
    f = BlackBoxOptim.best_fitness(res)
    return @ntuple(θ, f)
end

function optim_solver(fg!, θ0; verbose = false, fixedsigma = false, precond = nothing)
    lo = fixedsigma ? zeros(length(θ0)) : push!(zeros(length(θ0)-1), -Inf)
    hi = fill(Inf, length(θ0))

    inner_alg_type = Optim.LBFGS
    inner_alg = isnothing(precond) ? inner_alg_type(m = 10) : inner_alg_type(m = 10, P = precond)
    # inner_alg_type = Optim.ConjugateGradient #Optim.GradientDescent()
    # inner_alg = isnothing(precond) ? inner_alg_type() : inner_alg_type(P = precond)
    alg = Optim.Fminbox(inner_alg)

    @timeit mle_timer "Optim" begin
        res = Optim.optimize(Optim.only_fg!(fg!), lo, hi, θ0, alg)
        verbose && println("Optim $(inner_alg_type):\n\tgot $(Optim.minimum(optres)) after $(Optim.iterations(optres)) iterations")
    end

    θ = Optim.minimizer(res)
    f = Optim.minimum(res)

    return @ntuple(θ, f)
end

function nlopt_solver(fg!, θ0; verbose = false, fixedsigma = false, precond = nothing)
    alg = nloptalg[]
    opt = NLopt.Opt(alg, length(θ0))
    opt.lower_bounds = fixedsigma ? zeros(length(θ0)) : push!(zeros(length(θ0)-1), -Inf)
    opt.xtol_rel = 1e-4
    # opt.ftol_rel = 1e-4
    opt.min_objective = (x::Vector, grad::Vector) -> length(grad) > 0 ? fg!(true, grad, x) : fg!(true, nothing, x)

    minx = copy(θ0)
    @timeit mle_timer "NLopt ($alg)" begin
        minf, minx, ret = NLopt.optimize!(opt, minx)
        verbose && println("NLopt $(opt.algorithm):\n\tgot $minf after $(opt.numevals) iterations (returned $ret)")
    end

    return (θ = minx, f = minf)
end

function loglikelihood_inference(
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64},
        initial_guess::Union{AbstractVector{Float64},Nothing} = nothing,
        λ::Union{Float64,Nothing} = nothing,
        sigma::Union{Float64,Nothing} = nothing,
        solver = nlopt_solver;
        verbose = false,
        precond = nothing,
    )
    mle_loss! = make_rician_mle_loss(A, b, λ, sigma)
    @unpack θ, f = solver(mle_loss!, initial_guess; verbose = verbose, fixedsigma = !isnothing(sigma), precond = precond)

    # x = exp.(θ[1:end-1]) #TODO x -> logx
    x = isnothing(sigma) ? θ[1:end-1] : copy(θ)
    ν = A * x
    σ = isnothing(sigma) ? exp(θ[end]) : sigma

    return @ntuple(θ, x, ν, σ, f)
end

foreach(k -> opt_failures[k] = 0, keys(opt_failures));
TimerOutputs.reset_timer!(mle_timer);
for I in [CartesianIndex(1,1,1)] #CartesianIndices((10, 1:size(res[:image],2), 1))
    @unpack nT2, nTE, T2Range = default_t2mapopts
    T2_times = DECAES.logrange(T2Range..., nT2)
    nnls_t2dist = res[:dist][I,:]
    noisy_signal = res[:image][I,:]
    μ = res[:maps]["mu"][I] # regularization param from NNLS
    # decay_basis = res[:maps]["decaybasis"][I,:,:]
    decay_basis = zeros(nTE, nT2)
    DECAES.epg_decay_basis!(DECAES.EPGdecaycurve_work(Float64, nTE), decay_basis, default_flip_angle, T2_times, default_t2mapopts)

    σ0 = std(decay_basis * nnls_t2dist - noisy_signal)
    # lambda = μ^2/σ0^2 #nothing
    T2spacing, T2maxspacing = (T2Range[2]/T2Range[1])^(1/(nT2-1)), 1.5
    nwidth = log(T2maxspacing)/log(T2spacing)
    lambda = inv(maximum(noisy_signal)/nwidth)^2
    sigma = σ0
    precond = nothing #cholesky(inv(σ0^2) * (decay_basis' * decay_basis) + lambda * LinearAlgebra.I)
    θ0 = max.(nnls_t2dist, 1e-3*maximum(nnls_t2dist)) # init with NNLS solution + fixed sigma
    # sigma = nothing
    # θ0 = [max.(nnls_t2dist, 1e-3*maximum(nnls_t2dist)); log(σ0)] # init with NNLS solution

    # foreach(k -> opt_failures[k] = 0, keys(opt_failures))
    out = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda, sigma; verbose = true, precond = precond);

    # TimerOutputs.reset_timer!(mle_timer)
    for i in 1:5
        optres[:Optim] = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda, sigma, optim_solver, precond = precond);
        for alg in nloptalgs
            nloptalg[] = alg
            optres[alg] = loglikelihood_inference(decay_basis, noisy_signal, θ0, lambda, sigma, nlopt_solver, precond = precond)
        end
    end
    TimerOutputs.print_timer(mle_timer)
    println("")

    bestf = minimum(res -> res.f, values(optres))
    foreach(optres) do (alg, res)
        (res.f > bestf + 0.1) && (opt_failures[alg] += 1)
    end
    display(opt_failures)

    t2dist = out.x
    sigma = out.σ
    signal = out.ν
    SNR = -20 * log10(sigma / signal[1])
    plot(T2_times, [t2dist nnls_t2dist]; lab = ["mle" "nnls"], xscale = :log10, xformatter = x->string(round(x;sigdigits=3))) |> display
    @show lambda, σ0, sigma, SNR
    @show sqrt(1/lambda), maximum(noisy_signal)
    println("nnls_t2dist:"); display(T2partSEcorr(reshape(nnls_t2dist,1,1,1,:), default_t2partopts))
    println("t2dist:"); display(T2partSEcorr(reshape(t2dist,1,1,1,:), default_t2partopts))
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
