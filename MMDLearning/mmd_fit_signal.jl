# Load files
include(joinpath(@__DIR__, "src", "mmd_preamble.jl"))
include(joinpath(@__DIR__, "src", "mmd_math.jl"))
include(joinpath(@__DIR__, "src", "mmd_utils.jl"))

####
#### Signal fitting
####

function test_signal_fit(signal::AbstractVector{T};
        TE = T(opts.TE),
        T2Range = T.(opts.T2Range),
        nT2 = opts.nT2::Int,
        nTE = opts.nTE::Int,
        ntheta = settings["data"]["ntheta"]::Int,
        theta_labels = settings["data"]["theta_labels"]::Vector{String},
        kwargs...
    ) where {T}
    t2bins = T.(1000 .* DECAES.logrange(T2Range..., nT2)) # milliseconds
    t2times = T.(1000 .* TE .* (1:nTE)) # milliseconds
    xdata, ydata = t2times, signal./sum(signal)

    count = Ref(0)
    work = signal_model_work(T)
    model = θ -> (count[] += 1; return signal_model!(work, θ, zeros(T,1)))
    loss = θ -> (work.buf .= ydata .- model(θ); return sum(abs2, work.buf))
    res = bboptimize(loss;
        SearchRange = theta_bounds(T),
        TraceMode = :silent,
        MaxTime = T(1.0),
        MaxFuncEvals = typemax(Int),
        kwargs...
    )

    ϵ = zeros(T,1)
    θ = best_candidate(res)
    ℓ = best_fitness(res)
    ymodel = model(θ)

    p = plot(xdata, [ydata ymodel];
        title = "fitness = $(round(ℓ; sigdigits = 3))",
        label = ["data" "model"],
        annotate = (t2times[3*(end÷4)], 0.5*maximum(ymodel), join(theta_labels .* " = " .* string.(round.(θ; sigdigits=4)), "\n")),
        xticks = t2times, xrotation = 90)
    display(p)

    return θ, ϵ, ymodel
end
test_signal_fit(image::Array{T,4}, I::CartesianIndex; kwargs...) where {T} = test_signal_fit(image[I,:]; kwargs...)

function make_save_dict(res)
    # Results are BlackBoxOptim.OptimizationResults
    save_dict = Dict{String, Any}(
        "best_candidate"   => best_candidate(res),
        "best_fitness"     => best_fitness(res),
        "elapsed_time"     => res.elapsed_time,
        "f_calls"          => res.f_calls,
        "fit_scheme"       => string(res.fit_scheme),
        "iterations"       => res.iterations,
        "method"           => string(res.method),
        "start_time"       => res.start_time,
        "stop_reason"      => res.stop_reason,
        # "archive_output" => res.archive_output,
        # "method_output"  => res.method_output,
        # "parameters"     => res.parameters,
    )
    save_dict["settings"] = deepcopy(settings) #Dict(string.(keys(settings)) .=> deepcopy(values(settings)))
    # save_dict["opts"] = fieldnames(typeof(opts)) |> f -> Dict(string.(f) .=> getfield.(Ref(opts), f))
    return save_dict
end

####
#### Signal fitting
####

#=
for _ in 1:10
    σ = exp(settings["prior"]["logsigma"]::Float64)
    n = settings["prior"]["batchsize"]
    mmd_heatmap(signal_data(image, n), signal_data(image, n), σ) |> display
end
=#

#=
let
    t2bins = 1000 .* DECAES.logrange(opts.T2Range..., opts.nT2) # milliseconds
    t2times = 8 .* (1:settings["data"]["nsignal"]::Int) # milliseconds
    # heatmap(t2parts["sfr"][:,:,24]; xticks = 5:5:240, yticks = 5:5:240, xrotation = 90)
    plot(t2bins, t2dist[idx,:]; title = "t2dist: index = $(Tuple(idx))", xticks = t2bins, xrotation = 90, xscale = :log10, xformatter = x -> string(round(x; sigdigits=3))) |> display
    plot(t2times, image[idx,:]; title = "signal: index = $(Tuple(idx))", xticks = t2times, xrotation = 90) |> display
end
=#

#=
let
    # idx = CartesianIndex(141,144,24)
    # idx = CartesianIndex(150,111,24)
    # idx = CartesianIndex(154,80,24)
    # idx = CartesianIndex(111,127,24)
    idx = CartesianIndex(133,127,24)
    test_signal_fit(image, idx)[1]'
end
=#

# Y = signal_data(image, settings["prior"]["nbatches"]::Int)
# out = [test_signal_fit(Y[:,j]; MaxTime = settings["prior"]["maxtime"]::Float64) for j in 1:size(Y,2)] #@show(j)
# thetas = reduce(hcat, (x->x[1]).(out))
# noise  = reduce(hcat, (x->x[2]).(out))
# ymodel = reduce(hcat, (x->x[3]).(out))

# save_dict = Dict{String, Any}(
#     "thetas"   => copy(thetas),
#     "noise"    => copy(noise),
#     "ymodel"   => copy(ymodel),
#     "ydata"    => copy(Y),
#     "settings" => deepcopy(settings), #Dict(string.(keys(settings)) .=> deepcopy(values(settings))),
# )
# DECAES.MAT.matwrite(joinpath(settings["data"]["out"], "bbsignalfit_results.mat"), save_dict)

# phist = plot(map((lab,row,xl) -> histogram(row; nbins = 50, yticks = :none, lab = lab, xlims = xl), settings["data"]["theta_labels"], eachrow(thetas), theta_bounds())...; xrot = 45); display(phist)
# map(ext -> savefig(phist, joinpath(settings["data"]["out"], "thetas_hist.$ext")), ["pdf", "png"])

# pheat = mmd_heatmap(ymodel[:,1:min(50,end)], Y[:,1:min(50,end)], exp(settings["prior"]["logsigma"])); display(pheat)
# map(ext -> savefig(pheat, joinpath(settings["data"]["out"], "mmd_heatmap.$ext")), ["pdf", "png"])

#=
let
    p = plot(;leg = :none)
    plot!(sampleY(); c = :blue)
    plot!(sampleX(); c = :red)
    display(p)
    display(mmd_heatmap(sampleX(), sampleY(), 0.01))
end
=#

nothing
