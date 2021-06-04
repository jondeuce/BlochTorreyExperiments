####
#### Generate simulated data.
####
####    Resulting data and parameters are stored in /data/simulated/{timestamp}.
####    Move {timestamp} folder to /data/images for use with the rest of the pipeline.
####

using DrWatson: @quickactivate
@quickactivate "NeurIPS2021"
using NeurIPS2021
lib.initenv()

function biexp_epg_workspace(; nsignal, TEimage)
    phys    = lib.BiexpEPGModel{Float32}(n = nsignal)
    @unpack TEbd, T2bd, T1bd = phys
    TE        = 1.0 # all times are relative to TEimage and unitless
    TEimage   = 10e-3
    T1        = 1.0 / TEimage |> Float64
    logτ2lo   = log(T2bd[1] / TEbd[2]) |> Float64
    logτ2hi   = log(T2bd[2] / TEbd[1]) |> Float64
    logτ1lo   = log(T1bd[1] / TEbd[2]) |> Float64
    logτ1hi   = log(T1bd[2] / TEbd[1]) |> Float64
    logτ2lo′  = log(T2bd[1] / TEimage) |> Float64
    logτ2hi′  = log(T2bd[2] / TEimage) |> Float64
    δ0        = (log(T1) - logτ1lo) / (logτ1hi - logτ1lo)
    epg_cache = lib.BiexpEPGModelWorkCache(nsignal)
    return (; TEbd, T2bd, T1bd, TE, T1, logτ2lo, logτ2hi, logτ1lo, logτ1hi, logτ2lo′, logτ2hi′, δ0, epg_cache)
end

function simulate_biexpepg(; nsamples = 10, nsignal = 64, TEimage = 10e-3, seed = 0)
    rng  = Random.MersenneTwister(seed)
    α    = rand(rng, Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0), nsamples)
    β    = rand(rng, Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0), nsamples)
    η    = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0), nsamples)
    δ1   = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0), nsamples)
    δ2   = rand(rng, Distributions.TruncatedNormal(1.0, 0.5, 0.0, 1.0), nsamples)
    logϵ = rand(rng, Distributions.Uniform(log(1e-5), log(1e-1)), nsamples) # equivalent to 20 <= SNR <= 100
    logs = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, -2.5, 2.5), nsamples)
    work = [biexp_epg_workspace(; nsignal, TEimage) for _ in 1:Threads.nthreads()]
    X    = zeros(nsignal, nsamples)
    Threads.@threads for j in 1:nsamples
        X[:,j] .= lib.biexp_epg_model_scaled!(work[Threads.threadid()], α[j], β[j], η[j], δ1[j], δ2[j], logϵ[j], logs[j])
    end
    X    = map(d -> rand(rng, d), lib.Rician.(X, exp.(logϵ'))) # add Rician noise
    Xmax = maximum(X; dims = 1)
    X    ./= Xmax # scale X to maximum value 1
    logs .-= log.(vec(Xmax)) # shift log scale accordingly
    X    = permutedims(reshape(X, nsignal, nsamples, 1, 1), (2, 3, 4, 1))
    θ    = DataFrame((; α, β, η, δ1, δ2, logϵ, logs))
    return θ, X
end

function main(; nsamples = 200_000, nsignal = 64, TEimage = 10e-3, seed = 0)
    # Simulate data
    θ, X = simulate_biexpepg(; nsamples, nsignal, TEimage, seed)

    # Save data and parameters
    savefolder = mkpath(DrWatson.datadir("simulated", lib.getnow()))
    mkpath(joinpath(savefolder, "data-in"))
    DECAES.MAT.matwrite(joinpath(savefolder, "data-in", "theta.mat"), Dict("alpha" => θ.α, "beta" => θ.β, "eta" => θ.η, "delta1" => θ.δ1, "delta2" => θ.δ2, "logepsilon" => θ.logϵ, "logscale" => θ.logs))
    DECAES.MAT.matwrite(joinpath(savefolder, "data-in", "simulated_image.mat"), Dict("data" => X))

    # Save image parameters
    open(joinpath(savefolder, "image_info.toml"); write = true) do io
        image_info = Dict(
            "echotime" => TEimage,
            "refcon" => 180.0,
            "image_data_path" => "data-in/simulated_image.mat",
            "true_labels_path" => "data-in/theta.mat",
        )
        TOML.print(io, image_info)
    end

    # Save library code for future reference
    lib.save_project_code(joinpath(savefolder, "project"))

    return nothing
end

main(;
    nsamples = 200_000,
    nsignal  = 64,
    TEimage  = 10e-3,
    seed     = 0,
)
