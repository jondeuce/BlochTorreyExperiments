using DrWatson: @quickactivate
@quickactivate "Playground"
using Playground
lib.initenv()

function make_mcmc_biexp_epg_work(; nsignal, TEimage)
    phys    = lib.EPGModel{Float32,true}(n = nsignal)
    @unpack TEbd, T2bd, T1bd = phys

    TE        = 1.0 # all times are relative to TE and unitless
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

function make_mock_image_data(; nsamples = 10, nsignal = 64, TEimage = 10e-3, seed = 0)
    rng  = Random.MersenneTwister(seed)
    α    = rand(rng, Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0), nsamples)
    β    = rand(rng, Distributions.TruncatedNormal(180.0, 45.0, 90.0, 180.0), nsamples)
    η    = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0), nsamples)
    δ1   = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, 0.0, 1.0), nsamples)
    δ2   = rand(rng, Distributions.TruncatedNormal(1.0, 0.5, 0.0, 1.0), nsamples)
    logϵ = rand(rng, Distributions.Uniform(log(1e-5), log(1e-1)), nsamples) # equivalent to 20 <= SNR <= 100
    logs = rand(rng, Distributions.TruncatedNormal(0.0, 0.5, -2.5, 2.5), nsamples)
    work = [make_mcmc_biexp_epg_work(; nsignal, TEimage) for _ in 1:Threads.nthreads()]
    X    = zeros(nsignal, nsamples)
    Threads.@threads for j in 1:nsamples
        X[:,j] .= lib.biexp_epg_model_scaled!(work[Threads.threadid()], α[j], β[j], η[j], δ1[j], δ2[j], logϵ[j], logs[j])
    end
    X̂    = map(d -> rand(rng, d), lib.Rician.(X, exp.(logϵ')))
    X̂max = maximum(X̂; dims = 1)
    X̂    ./= X̂max # scale X̂ to maximum value 1
    logs .-= log.(vec(X̂max)) # shift log scale accordingly
    X̂    = permutedims(reshape(X̂, nsignal, nsamples, 1, 1), (2, 3, 4, 1))
    θ    = DataFrame((; α, β, η, δ1, δ2, logϵ, logs))
    return θ, X̂
end

function make_mock_phys(; nsamples = 200_000, nsignal = 64, TEimage = 10e-3, seed = 0)
    θ, X̂ = make_mock_image_data(; nsamples, nsignal, TEimage, seed)
    CSV.write("theta.csv", θ)
    DECAES.MAT.matwrite("simulated_image.mat", Dict("data" => X̂))
    phys = lib.EPGModel{Float32,false}(n = nsignal)
    image_infos = [Dict("echotime" => TEimage, "refcon" => 180.0, "image_data_path" => "./simulated_image.mat")]
    lib.initialize!(phys; image_infos, seed)
    return phys
end

# Save library code for future reference
lib.save_project_code(joinpath(pwd(), "project"))

# Perform mcmc
phys = make_mock_phys(;
    nsamples    = 200_000,
    nsignal     = 64,
    TEimage     = 10e-3,
    seed        = 0,
)
for dataset in [:val, :test, :train]
    lib.mcmc_biexp_epg(
        phys;
        img_idx         = 1,
        dataset         = dataset,
        save            = true,
        checkpoint      = true,
        total_chains    = Colon(),
        checkpoint_freq = 2048, # checkpoint every `checkpoint_freq` iterations
        progress_freq   = 15.0, # update progress bar every `progress_freq` seconds
    )
end
