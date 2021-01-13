function Augmenter(;
        signal::Bool = false, # Plain signal
        gradient::Bool = false, # Signal gradient
        laplacian::Bool = false, # Signal laplacian
        fdcat::Tuple{Int,Int} = (0,0), # Signal finite differences, concatenated
        encoderspace::Bool = false, # Encoder-space signal
        fftcat::Bool = false, # Concatenated real/imag fourier components
        fftsplit::Bool = false, # Separate real/imag fourier components
    )

    aug_labels = []
    inner_vars = []
    aug_models = []

    if signal
        push!(aug_models, identity)
        push!(inner_vars, ("X₀",))
        push!(aug_labels, :signal)
    end
    if gradient
        push!(aug_models, ForwardDifferemce())
        push!(inner_vars, ("∇X",))
        push!(aug_labels, :grad)
    end
    if laplacian
        push!(aug_models, Laplacian())
        push!(inner_vars, ("∇²X",))
        push!(aug_labels, :lap)
    end
    if all(fdcat .> 0);
        push!(aug_models, CatFiniteDifference(fdcat...))
        push!(inner_vars, ("∇ⁿX",))
        push!(aug_labels, :fd)
    end
    if fftcat
        push!(aug_models, X -> vcat(reim(rfft(X,1))...))
        push!(inner_vars, ("ℱX",))
        push!(aug_labels, :fft)
    end
    if fftsplit
        push!(aug_models, X -> reim(rfft(X,1)))
        push!(inner_vars, ("ℜℱX", "ℑℱX"))
        append!(aug_labels, (:rfft, :ifft))
    end
    # if residuals
    #     push!(aug_models, identity) #TODO Xres = X .- Zygote.@ignore(sampleXθZ(derived["cvae"], derived["prior"], X; posterior_θ = true, posterior_Z = true))[1] : nothing # Residual relative to different sample X̄(θ), θ ~ P(θ|X) (note: Z discarded, posterior_Z irrelevant)
    #     push!(inner_vars, "Xres")
    #     push!(aug_labels, :res)
    # end
    # if encoderspace
    #     push!(aug_models, identity) #TODO Xenc = derived["encoderspace"](X) # Encoder-space signal
    #     push!(inner_vars, "Xenc")
    #     push!(aug_labels, :enc)
    # end

    topo_inner = join("X => (" .* join.(inner_vars, ",") .* ")", " : ")
    topo_str = "X : " * topo_inner * " : (" * join(foldl(append!!, inner_vars; init = []), ",") * ") => Xs"

    Stack(
        NNTopo(topo_str),
        aug_models...,
        CollectNamedTuple(aug_labels...),
    )
end
