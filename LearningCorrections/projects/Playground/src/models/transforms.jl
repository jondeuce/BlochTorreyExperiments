function ParallelCollector(
        models::AbstractVector,
        varname_list::AbstractVector,
        reducer = tuple,
    )
    all_varnames = foldl(append!!, varname_list; init = [])
    topo_inner = join("X => (" .* join.(varname_list, ",") .* ")", " : ")
    topo_str = "X : " * topo_inner * " : (" * join(foldl(append!!, varname_list; init = []), ",") * ") => Y"
    Stack(NNTopo(topo_str), models..., reducer)
end

function DomainTransforms(
        phys::PhysicsModel{Float32};
        signal::Bool = false, # Plain signal
        gradient::Bool = false, # Signal gradient
        laplacian::Bool = false, # Signal laplacian
        fdcat::Int = 0, # Signal finite differences, concatenated
        encoderspace::Bool = false, # Encoder-space signal
        fftcat::Bool = false, # Concatenated real/imag fourier components
        fftsplit::Bool = false, # Separate real/imag fourier components
        kwargs...,
    )

    models, varname_list, ntuple_labels = [], [], []
    function add!(lab, vars, m)
        models = push!!(models, ApplyAsMatrix(m))
        varname_list = push!!(varname_list, vars)
        ntuple_labels = append!!(ntuple_labels, lab)
    end

    signal        && add!((:signal,), ("X₀",), identity)
    gradient      && add!((:grad,), ("∇X",), ForwardDifference()) # DenseFiniteDiff(nsignal(phys), 1)
    laplacian     && add!((:lap,), ("∇²X",), Laplacian()) # DenseFiniteDiff(nsignal(phys), 2)
    fdcat > 0     && add!((:fd,), ("∇ⁿX",), CatDenseFiniteDiff(nsignal(phys), fdcat))
    fftcat        && add!((:fft,), ("ℱX",), X -> vcat(reim(rfft(X,1))...))
    fftsplit      && add!((:rfft, :ifft), ("ℜℱX", "ℑℱX"), X -> reim(rfft(X,1)))
    #residuals    && add!((:res,), ("Xres",), identity) #TODO X -> X .- Zygote.@ignore(sampleXθZ(phys, cvae, θprior, Zprior, X; posterior_θ = true, posterior_Z = true))[1] : nothing # Residual relative to different sample X̄(θ), θ ~ P(θ|X) (note: Z discarded, posterior_Z irrelevant)
    #encoderspace && add!((:enc,), ("Xenc",), identity) #TODO X -> encoderspace(X) # Encoder-space signal

    ParallelCollector(models, varname_list, CollectAsNamedTuple(ntuple_labels...))
end

function domain_transforms_outdims(phys::PhysicsModel{Float32}; kwargs...)
    X = zeros(Float32, nsignal(phys))
    Aug = DomainTransforms(phys; kwargs...)
    map(size, Aug(X))
end
domain_transforms_outlengths(phys::PhysicsModel{Float32}; kwargs...) = map(first, domain_transforms_outdims(phys; kwargs...))
domain_transforms_sum_outlengths(phys::PhysicsModel{Float32}; kwargs...) = sum(domain_transforms_outlengths(phys; kwargs...))

function Augmentations(;
        chunk::Int,
        flipsignals::Bool,
        kwargs...
    )
    if chunk > 0 || flipsignals
        augmentations_inner(Xs...) = augmentations(Xs...; chunk, flipsignals)
    else
        identity
    end
end

function augmentations(Xs::Union{<:Tuple, <:NamedTuple}; chunk::Int, flipsignals::Bool)
    n = size(first(Xs), 1) # all Xs of same domain are equal size in first dimension
    i = Zygote.@ignore begin
        i = ifelse(!(0 < chunk < n), 1:n, rand(1:n-chunk+1) .+ (0:chunk-1))
        i = ifelse(!flipsignals, i, ifelse(rand(Bool), i, reverse(i)))
    end
    Ys = map(X -> X[i,..], Xs) # tuple of transformed augmentations
end

function pad_and_mask_signal(Y::AbstractVecOrMat, n; minkept, maxkept)
    Ypad  = pad_signal(Y, n)
    M     = Zygote.@ignore signal_mask(Ypad; minkept, maxkept)
    Ymask = M .* Ypad
    return Ymask, M
end
