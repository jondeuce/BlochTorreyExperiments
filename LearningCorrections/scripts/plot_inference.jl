####
#### Code loading
####

using MMDLearning
pyplot(size = (800,600))

# Convenience functions
split_correction_and_noise(μlogσ) = μlogσ[1:end÷2, :], exp.(μlogσ[end÷2+1:end, :])
correction_and_noiselevel(G, X) = split_correction_and_noise(G(X))
get_rician_params(G, X) = ((dX, ϵ) = correction_and_noiselevel(G, X); return (abs.(X .+ dX), ϵ))

# Inference wrapper
function do_inference(F, P, Y, θ, θ0, θbd)
    inf_results = signal_loglikelihood_inference(Y, θ0, P, F; objective = :mle, bounds = θbd)
    if inf_results isa AbstractVector
        mle_results = map(inf_results) do (_, optim_res)
            (x = Optim.minimizer(optim_res), loss = Optim.minimum(optim_res))
        end
        logL = (r -> -r.loss).(mle_results)
        θhat = reduce(hcat, (r -> r.x).(mle_results))
    else
        _, optim_res = inf_results
        logL = -Optim.minimum(optim_res)
        θhat = Optim.minimizer(optim_res)
    end
    θerr = abs.(θ .- θhat)
    return @ntuple(Y, θ, θhat, θerr, logL)
end

# Print results
function print_inference(inf_res; θscale = 1, digits = 2)
    for (name, res) in inf_res
        s = x -> round(x; digits = digits)
        println("Model = $name:")
        println(" logL = $(s(mean(res.logL))) ± $(s(std(res.logL) / sqrt(size(res.Y,2))))")

        θmu = mean(θscale .* res.θerr; dims = 2)
        θstd = std(θscale .* res.θerr; dims = 2) ./ sqrt(size(res.θerr,2))
        for i in 1:length(θmu)
            println("   θ$i = $(s(θmu[i])) ± $(s(θstd[i]))")
        end
    end
    return inf_res
end

####
#### Toy model inference
####

const TOYMODELS = Dict{String,Any}(
    "CVAE" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/toycvae_eps=1e-2/toycvae-both-corr-v1/sweep/16/log/2020-05-02-T-15-26-00-569.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=4_Xout=5_Zdim=8_act=relu_boundmean=true.model-best.bson")[:model]),
    "MMD"  => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/toymmdopt-v9/sweep/15/best-model.bson")["mmd"]),
    "HYB"  => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-toyganopt-v5/sweep/78/best-model.bson")["G"]),
    "GAN"  => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/toyganopt-v3/sweep/24/best-model.bson")["G"]),
);

# Toy model inference
function do_toy_inference(nsamples = 1)
    Random.seed!(0)
    mock_forward_model = (θ, noise) -> toy_signal_model(θ, noise, 4)
    true_forward_model = (θ, noise) -> toy_signal_model(θ, noise, 2)

    ϵ0  = 0.01
    θbd = toy_theta_bounds()
    θ   = toy_theta_sampler(nsamples)
    Y   = true_forward_model(θ, ϵ0)

    @time θ0 = TOYMODELS["CVAE"](Y)
    results = Dict{String, Any}()

    # Fit F4 and F2 models
    F4 = θ -> mock_forward_model(θ, nothing)
    F2 = θ -> true_forward_model(θ, nothing)
    P  = ν -> (ν, ϵ0)
    @time results["F4"] = do_inference(F4, P, Y, θ, θ0, θbd)
    @time results["F2"] = do_inference(F2, P, Y, θ, θ0, θbd)

    # Compute CVAE models
    @time results["CVAE"] = (
        Y = Y, θ = θ, θhat = θ0,
        θerr = abs.(θ .- θ0),
        logL = vec(sum(logpdf.(Rician.(F2(θ0), ϵ0), Y); dims = 1)),
    )

    # Fit GAN models
    for name in ["MMD", "HYB", "GAN"]
        G = deepcopy(TOYMODELS[name])
        P = X -> get_rician_params(G, X)
        @time results[name] = do_inference(F4, P, Y, θ, θ0, θbd)
    end

    return results
end

@time do_toy_inference(2);
@time toy_inf = print_inference(do_toy_inference(1000); θscale = (bd -> 100 / (bd[2]-bd[1])).(toy_theta_bounds()));

BSON.bson("toy_inference.bson", Dict("toy_models" => deepcopy(TOYMODELS), "toy_inference" => deepcopy(toy_inf)));

####
#### MRI model inference
####

# Load data samplers
#=
=#
const mri_settings = load_settings("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/src/hybrid_settings.toml");
sampleX, sampleY, sampleθ, fits_train, fits_test, fits_val = make_mle_data_samplers(
    mri_settings["prior"]["data"]["image"]::String,
    mri_settings["prior"]["data"]["thetas"]::String;
    ntheta = mri_settings["data"]["ntheta"]::Int,
    padtrain = false,
    normalizesignals = false,
    filteroutliers = false,
    plothist = false,
);

# Toy model inference
function do_mri_inference(mrimodels; nsamples = 1, seed = 0)
    # CVAE params:   cosd(alpha), T2short, T2long, Ashort, Along
    # Signal params:       alpha, T2short,    dT2, Ashort, Along
    Random.seed!(seed)
    Is   = sample(MersenneTwister(seed), 1:size(sampleY(nothing; dataset = :test), 2), nsamples; replace = false)
    θ    = sampleθ(nothing; dataset = :test)[:, Is]
    X    = sampleX(nothing; dataset = :test)[:, Is]
    Y    = sampleY(nothing; dataset = :test)[:, Is]
    fits = deepcopy(fits_val)[Is, :]
    θ0   = deepcopy(θ)
    θbd  = vec(extrema(sampleθ(nothing; dataset = :test); dims = 2))
    #θbd = [(50.0, 180.0), (8.0, 1000.0), (8.0, 1000.0), (0.0, 3.8), (0.0, 21.0)]
    results = Dict{String, Any}()

    # Fit mri signal model
    nTE = mri_settings["data"]["nsignal"]::Int
    TE  = mri_settings["data"]["echotime"]::Float64
    F   = θ -> signal_model(θ, nothing; nTE = nTE, TE = TE, normalize = false)
    results["F"] = (Y = Y, θ = θ, θhat = θ, θerr = zero(θ), logL = -fits.loss)
    # @time results["F"] = map(1:nsamples) do j
    #     Threads.@spawn begin
    #         ϵ0 = exp.(fits.logsigma[j])
    #         P  = ν -> (ν, ϵ0)
    #         do_inference(F, P, Y[:,j], θ[:,j], θ0[:,j], θbd)
    #     end
    # end |> r -> map(Threads.fetch, r) |> r -> (
    #     Y    = reduce(hcat, [r.Y for r in r]),
    #     θ    = reduce(hcat, [r.θ for r in r]),
    #     θhat = reduce(hcat, [r.θhat for r in r]),
    #     θerr = reduce(hcat, [r.θerr for r in r]),
    #     logL = (r -> r.logL).(r),
    # )

    # Compute CVAE models
    # @time θ0_cvae = mrimodels["CVAE"](Y)
    # θ0_cvae[1,:] .= acosd.(clamp.(θ0_cvae[1,:], -1, 1)); # transform cosd(flipangle) -> flipangle
    # θ0_cvae[3,:] .= θ[3,:] .- θ[2,:]; # transform T2long = T2short + dT2 -> dT2
    # θ0_cvae .= ((θ, bd) -> clamp(θ, bd...)).(θ0_cvae, θbd)
    # @time nu_cvae, ϵ_cvae = get_rician_params(mrimodels["CVAE-MMD"], F(θ0_cvae))
    # @time nu_cvae, ϵ_cvae = get_rician_params(mrimodels["MMD"], F(θ0_cvae))
    # @time results["CVAE"] = (
    #     Y = Y, θ = θ, θhat = θ0_cvae,
    #     θerr = abs.(θ .- θ0_cvae),
    #     logL = vec(sum(logpdf.(Rician.(nu_cvae, ϵ_cvae), Y); dims = 1)),
    # )

    # Fit GAN models
    # for (name, G) in mrimodels
    #     startswith(name, "CVAE") && continue
    #     G = deepcopy(mrimodels[name])
    #     P = X -> get_rician_params(G, X)
    #     print("$name:")
    #     @time results[name] = do_inference(F, P, Y, θ, θ0, θbd)
    # end

    print("MMD:"); G = deepcopy(mrimodels["MMD"]); P = X -> get_rician_params(G, X); @time results["MMD"] = do_inference(F, P, Y, θ, θ0, θbd)
    print("GAN:"); G = deepcopy(mrimodels["GAN"]); P = X -> get_rician_params(G, X); @time results["GAN"] = do_inference(F, P, Y, θ, copy(results["MMD"].θhat), θbd)
    print("HYB:"); G = deepcopy(mrimodels["HYB"]); P = X -> get_rician_params(G, X); @time results["HYB"] = do_inference(F, P, Y, θ, copy(results["MMD"].θhat), θbd)

    return results
end

const MRIMODELS = Dict{String,Any}(
    # "CVAE" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-train-noise-test-corr-v1/sweep/6/log/2020-05-01-T-12-22-13-146.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=32_Nh=6_Xout=5_Zdim=8_act=relu_boundmean=true.model-best.bson")[:model]), # Trained on uncorrected signal + MMD-learned noise
    "CVAE" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-both-corr-v2/sweep/35/log/2020-05-01-T-12-05-36-268.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=6_Xout=5_Zdim=6_act=relu_boundmean=true.model-best.bson")[:model]), # Trained on MMD-corrected signal + MMD-learned noise
    # "CVAE-MMD" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/cvae-both-corr-v2/sweep/35/log/2020-05-01-T-12-05-36-268.acc=rmse_gamma=1_loss=l2_DenseLIGOCVAE_Dh=128_Nh=6_Xout=5_Zdim=6_act=relu_boundmean=true.mmd-models.bson")["mmd"]),
    "MMD" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/MMD-Learning/mmdopt-v7/sweep/126/best-model.bson")["mmd"]),
    "HYB" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/187/current-model.bson")["G"]),
    # "HYB" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v3/sweep/272/current-model.bson")["G"]),
    # "HYB" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/hybrid-mri-gan-opt-v2/sweep/90/best-model.bson")["G"]),
    "GAN" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/simulations/MMD-Learning/ganopt-v3/sweep/48/best-model.bson")["G"]),
    # "MMD" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/23-best-mmd-2020-05-30-T-12-03-23-765/plot-models.bson")["mmd"]["models"]["mmd"]),
    # "HYB" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/23-best-mmd-2020-05-30-T-12-03-23-765/plot-models.bson")["hyb"]["models"]["G"]),
    # "GAN" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/23-best-mmd-2020-05-30-T-12-03-23-765/plot-models.bson")["gan"]["models"]["G"]),
    # "MMD" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/69-best-tstat-2020-05-30-T-11-53-14-042/plot-models.bson")["mmd"]["models"]["mmd"]),
    # "HYB" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/69-best-tstat-2020-05-30-T-11-53-14-042/plot-models.bson")["hyb"]["models"]["G"]),
    # "GAN" => deepcopy(BSON.load("/project/st-arausch-1/jcd1994/code/BlochTorreyExperiments-active/MMDLearning/output/plots-checkpoint-4/69-best-tstat-2020-05-30-T-11-53-14-042/plot-models.bson")["gan"]["models"]["G"]),
);

@time do_mri_inference(MRIMODELS; nsamples = 2, seed = 0);
@time mri_inf = print_inference(do_mri_inference(MRIMODELS; nsamples = 32, seed = 0));

BSON.bson("mri_inference.bson", Dict("mri_models" => deepcopy(MRIMODELS), "mri_inference" => deepcopy(mri_inf)));

#=
# all_mri_inf = Dict{String,Any}[deepcopy(mri_inf)];
let j = 49
    global all_mri_inf
    while true
        j += 1
        println("\n\nBatch $j:")
        @time _mri_inf = print_inference(do_mri_inference(MRIMODELS; nsamples = 32, seed = j));
        push!(all_mri_inf, _mri_inf)
    end
end

all_mri_inf_flat = let rs = deepcopy(all_mri_inf)
    r = deepcopy(rs[1])
    for j in 2:length(rs), name in keys(r)
        r[name] = (
            Y    = hcat(r[name].Y, rs[j][name].Y),
            θ    = hcat(r[name].θ, rs[j][name].θ),
            θhat = hcat(r[name].θhat, rs[j][name].θhat),
            θerr = hcat(r[name].θerr, rs[j][name].θerr),
            logL = vcat(r[name].logL, rs[j][name].logL),
        )
    end
    Is = hcat([sortperm(r[G].logL) for G in keys(r)]...) |>
        x -> intersect([c[50:end] for c in eachcol(x)]...) |>
        x -> x[end-1000+1:end]
    for name in keys(r)
        r[name] = (
            Y    = r[name].Y[:,Is],
            θ    = r[name].θ[:,Is],
            θhat = r[name].θhat[:,Is],
            θerr = r[name].θerr[:,Is],
            logL = r[name].logL[Is],
        )
    end
    r
end;
=#

#=
Model = HYB:
 logL = 175.4285188069668 ± 0.511360818055652
   θ1 = 3.630240379687224 ± 0.15687236860096263
   θ2 = 3.97736991080537 ± 0.1573048027974183
   θ3 = 63.9995359203556 ± 4.881924095359591
   θ4 = 0.12584269532070305 ± 0.007785646061124261
   θ5 = 0.13088415993888505 ± 0.007946825792797683
Model = GAN:
 logL = 173.99155511076268 ± 0.5547167008013014
   θ1 = 4.952115849478641 ± 0.21453117515512113
   θ2 = 4.4136150301951576 ± 0.1644055505364917
   θ3 = 61.85917258813918 ± 4.5000556587074385
   θ4 = 0.13296934490506218 ± 0.006843572141865662
   θ5 = 0.13371969225637823 ± 0.007171822567334387
Model = F:
 logL = 163.67002526697442 ± 0.5612748432075817
   θ1 = 0.0 ± 0.0
   θ2 = 0.0 ± 0.0
   θ3 = 0.0 ± 0.0
   θ4 = 0.0 ± 0.0
   θ5 = 0.0 ± 0.0
Model = MMD:
 logL = 173.99101623210382 ± 0.48495493885922375
   θ1 = 2.798188796906574 ± 0.10755780547212121
   θ2 = 3.249636243363402 ± 0.12515303868192632
   θ3 = 54.87404211380053 ± 4.465075125758724
   θ4 = 0.11001071673404335 ± 0.006349678501209169
   θ5 = 0.11600436099536918 ± 0.006429144078783687
=#

####
#### DECAES inference
####

decaes_inf = let
    global mri_inf
    local d = Dict{String,Any}()
    local Y = copy(mri_inf["F"].Y)
    local t2mapopts = T2mapOptions(
        MatrixSize = (size(Y,2), 1, 1),
        nTE        = size(Y,1),
        TE         = 8e-3,
        nT2        = 40,
        T2Range    = (8e-3, 1.0), #TODO range changed from (15e-3, 2.0) to match CVAE outputs
        Threshold  = 0.0, #TODO important since signals are "normalized" down uniformly by 1e6
        Silent     = true,
    );
    local t2partopts = T2partOptions(
        MatrixSize = (size(Y,2), 1, 1),
        nT2        = t2mapopts.nT2,
        T2Range    = t2mapopts.T2Range,
        SPWin      = (prevfloat(t2mapopts.T2Range[1]), 40e-3), #TODO
        MPWin      = (nextfloat(40e-3), nextfloat(t2mapopts.T2Range[2])), #TODO
        Silent     = true,
    );
    @time begin
        d["t2maps"], d["t2dist"] = DECAES.T2mapSEcorr(Array(reshape(permutedims(Y), (size(Y,2), 1, 1, size(Y,1)))), t2mapopts)
        d["t2parts"] = DECAES.T2partSEcorr(d["t2dist"], t2partopts)
    end
    d
end

function compare_decaes(mri_inf, decaes_inf)
    for (name, res) in mri_inf
        alpha   = mri_inf[name].θhat[1,:]
        T2short = mri_inf[name].θhat[2,:]
        dT2     = mri_inf[name].θhat[3,:]
        Ashort  = mri_inf[name].θhat[4,:]
        Along   = mri_inf[name].θhat[5,:]
        T2long  = T2short + dT2
        ggm     = @. exp((Ashort * log(T2short) + Along * log(T2long)) / (Ashort + Along))
        α_err   = abs.(alpha .- decaes_inf["t2maps"]["alpha"][:])
        ggm_err = abs.(ggm .- 1000 .* decaes_inf["t2maps"]["ggm"][:])
        println("$name:")
        println("    α: $(round(mean(α_err); digits = 2)) +/- $(round(std(α_err) / sqrt(length(α_err)); digits = 2))")
        println("  ggm: $(round(mean(ggm_err); digits = 2)) +/- $(round(std(ggm_err) / sqrt(length(ggm_err)); digits = 2))")
    end
end
compare_decaes(mri_inf, decaes_inf)

#=
mri_inf["CVAE"] = let
    global mri_inf
    local Y   = copy(mri_inf["F"].Y)
    local θ   = copy(mri_inf["F"].θ)
    local θbd = vec(extrema(sampleθ(nothing; dataset = :test); dims = 2))
    @time θ0_cvae = mean([MRIMODELS["CVAE"](Y) for _ in 1:100])
    θ0_cvae[1,:] .= acosd.(clamp.(θ0_cvae[1,:], -1, 1)); # transform cosd(flipangle) -> flipangle
    θ0_cvae[3,:] .= θ[3,:] .- θ[2,:]; # transform T2long = T2short + dT2 -> dT2
    θ0_cvae .= ((θ, bd) -> clamp(θ, bd...)).(θ0_cvae, θbd)
    (
        Y = Y, θ = θ, θhat = θ0_cvae,
        θerr = abs.(θ .- θ0_cvae),
        logL = fill(-Inf, size(θ0_cvae, 2)),
    )
end
=#