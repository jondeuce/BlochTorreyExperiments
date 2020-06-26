@with_kw struct DataInfo{T}
    nfeatures::Int = 32
    nchannels::Int = 1
    nlabels::Int = 8
    labmean::Vector{T} = zeros(nlabels)
    labwidth::Vector{T} = ones(nlabels)
end

function make_model(settings::Dict, name::String)
    if name == "load"
        return BSON.load(settings["model"][name]["path"])[:model]
    else
        T = settings["prec"] == 64 ? Float64 : Float32
        model_maker = eval(Symbol(name))
        kwargs = make_model_kwargs(T, deepcopy(settings["model"][name]))
        info = DataInfo{T}(; [f => deepcopy(settings["data"]["info"][string(f)]) for f in fieldnames(DataInfo)]...)
        model = model_maker(info; kwargs...)
        return Flux.paramtype(T, model) # ensure correct underlying float type
    end
end
make_model(settings::Dict) = Dict{String,Any}([name => make_model(settings, name) for name in keys(settings["model"]) if is_model_name(name)]...)

function is_model_name(name::Symbol)
    try
        hasmethod(MMDLearning.eval(name), (DataInfo,))
    catch e
        (e isa UndefVarError) ? false : rethrow(e)
    end
end
is_model_name(name::String) = is_model_name(Symbol(name))

function model_string(settings::Dict, name::String)
    if name == "load"
        mstring = splitext(basename(settings["model"][name]["path"]))[1]
        dateregex = r"\d\d\d\d-\d\d-\d\d-T-\d\d-\d\d-\d\d-\d\d\d."
        endswith(mstring, ".model-best") && (mstring = mstring[1:end-11])
        endswith(mstring, ".model-checkpoint") && (mstring = mstring[1:end-17])
        (match(dateregex, mstring, 1) != nothing) && (mstring = mstring[27:end])
        mstring
    else
        # Enumerated and replace vector properties 
        d = deepcopy(settings["model"][name])
        for (k,v) in deepcopy(d)
            if v isa AbstractVector
                for i in 1:length(v)
                    d[k * string(i)] = v[i]
                end
                delete!(d, k)
            end
        end
        mstring = name * "_" * DrWatson.savename(d)
    end
end
model_string(settings::Dict) = DrWatson.savename(settings["model"]) * "_" * join([model_string(settings, name) for (name, model) in settings["model"] if is_model_name(name)], "_")

make_model_kwargs(::Type{T}, m::Dict) where {T} = Dict{Symbol,Any}(Symbol.(keys(m)) .=> clean_model_arg.(T, values(m)))
clean_model_arg(::Type{T}, x) where {T} = error("Unsupported model parameter type $(typeof(x)): $x") # fallback
clean_model_arg(::Type{T}, x::Bool) where {T} = x
clean_model_arg(::Type{T}, x::Integer) where {T} = Int(x)
clean_model_arg(::Type{T}, x::AbstractString) where {T} = Symbol(x)
clean_model_arg(::Type{T}, x::AbstractFloat) where {T} = T(x)
clean_model_arg(::Type{T}, x::AbstractArray) where {T} = convert(Vector, deepcopy(vec(clean_model_arg.(T, x))))

make_activation(name::Symbol) = Flux.eval(name)
make_activation(name::String) = make_activation(Symbol(name))

"""
    Sequence classification with 1D convolutions:
        https://keras.io/getting-started/sequential-model-guide/
"""
function Keras1DSeqClass(
        info    :: DataInfo{T} = DataInfo();
        Nf1     :: Int = 4,
        Nf2     :: Int = 4,
        Npool   :: Int = 2,
        Nkern   :: Int = 3,
        act     :: Symbol = :leakyrelu,
        dropout :: T = T(0.0),
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H = nfeatures # data height
    C = nchannels # number of channels
    Nout = nlabels # output size

    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf2 * (H ÷ Npool ÷ Npool)
    actfun = make_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling
        XavierConv((Nkern,1), C => Nf1, pad = (Npad,0), actfun), # (H, 1, 1) -> (H, Nf1, 1)
        XavierConv((Nkern,1), Nf1 => Nf1, pad = (Npad,0), actfun), # (H, Nf1, 1) -> (H, Nf1, 1)
        Flux.MaxPool((Npool,)), # (H, Nf1, 1) -> (H/Npool, Nf1, 1)

        # Two more convolution layers followed by mean pooling
        XavierConv((Nkern,1), Nf1 => Nf2, pad = (Npad,0), actfun), # (H/Npool, Nf1, 1) -> (H/Npool, Nf2, 1)
        XavierConv((Nkern,1), Nf2 => Nf2, pad = (Npad,0), actfun), # (H/Npool, Nf2, 1) -> (H/Npool, Nf2, 1)
        Flux.MeanPool((Npool,)), # (H/Npool, Nf2, 1) -> (H/Npool^2, Nf2, 1)

        # Dropout layer
        (dropout > 0) ? Flux.Dropout(dropout) : identity,

        # Dense + softmax layer
        DenseResize(),
        Flux.Dense(Ndense, Nout),
    )
    
    return model |> m -> Flux.paramtype(T, m)
end

function TestModel1(
        info      :: DataInfo{T} = DataInfo();
        Nf1       :: Int = 4,
        Nf2       :: Int = 4,
        Nf3       :: Int = 4,
        Npool     :: Int = 2,
        Nkern     :: Int = 3,
        act       :: Symbol = :leakyrelu,
        dropout   :: T = T(0.0),
        batchnorm :: Bool = true,
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H = nfeatures # data height
    C = nchannels # number of channels
    Nout = nlabels # output size

    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf3 * (H ÷ Npool ÷ Npool ÷ Npool)
    actfun = make_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling and batch normalization
        XavierConv((Nkern,1), C => Nf1, pad = (Npad,0), actfun),
        XavierConv((Nkern,1), Nf1 => Nf1, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        batchnorm ? Flux.BatchNorm(Nf1, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        XavierConv((Nkern,1), Nf1 => Nf2, pad = (Npad,0), actfun),
        XavierConv((Nkern,1), Nf2 => Nf2, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        batchnorm ? Flux.BatchNorm(Nf2, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        XavierConv((Nkern,1), Nf2 => Nf3, pad = (Npad,0), actfun),
        XavierConv((Nkern,1), Nf3 => Nf3, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        batchnorm ? Flux.BatchNorm(Nf3, actfun) : identity,

        # Dropout layer
        (dropout > 0) ? Flux.Dropout(dropout) : identity,

        # Dense / batchnorm layer
        DenseResize(),
        Flux.Dense(Ndense, Ndense ÷ 2, actfun),
        batchnorm ? Flux.BatchNorm(Ndense ÷ 2, actfun) : identity,

        # Dense / batchnorm layer, but last actfun must be softplus since outputs are positive
        batchnorm ? Flux.Dense(Ndense ÷ 2, Nout, actfun) : Flux.Dense(Ndense ÷ 2, Nout, NNlib.softplus),
        batchnorm ? Flux.BatchNorm(Nout, NNlib.softplus) : identity,

        # Scale from (0,1) back to model parameter range #TODO FIXME
        Scale(labwidth, labmean),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function TestModel2(
        info      :: DataInfo{T} = DataInfo();
        Nh        :: Int = 2,
        Dh        :: Int = info.nfeatures,
        act       :: Symbol = :leakyrelu,
        dropout   :: T = T(0.0),
        batchnorm :: Bool = true,
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H = nfeatures # data height
    C = nchannels # number of channels
    Nout = nlabels # output size
    actfun = make_activation(act)

    model = Flux.Chain(
        DenseResize(),
        Flux.Dense(H * C, Dh, actfun),
        # batchnorm ? Flux.BatchNorm(Dh, actfun) : identity,
        
        mapreduce(vcat, 1:Nh; init = []) do _ 
            [Flux.Dense(Dh, Dh, actfun)]
            # batchnorm ? [Flux.BatchNorm(Dh, actfun)] : []
        end...,

        Flux.Dense(Dh, Nout, actfun),
        batchnorm ? Flux.BatchNorm(Nout, actfun) : identity,

        # Softplus to ensure positivity
        @λ(x -> NNlib.softplus.(x)),

        # Scale from (0,1) back to model parameter range #TODO FIXME
        Scale(labwidth, labmean),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function TestModel3(
        info      :: DataInfo{T} = DataInfo();
        act       :: Symbol = :leakyrelu,
        dropout   :: T = T(0.0),
        batchnorm :: Bool = true,
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H      = nfeatures # Data height
    C      = nchannels # Number of channels
    Nout   = nlabels # output size
    actfun = make_activation(act)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Scale(labwidth, labmean)
    MakeDropout() = dropout > 0 ? Flux.Dropout(dropout) : identity
    ResidualBlock() = IdentitySkip(BatchDenseConnection(4, 4, actfun; groupnorm = batchnorm, mode = :post))

    # Output (assumes 1:2 is mwf/iewf and 3:4 are other physical params)
    model = Flux.Chain(
        DenseResize(),
        Flux.Dense(H*C, 16, actfun),
        ChannelResize(4),
        (ResidualBlock() for _ in 1:3)...,
        MakeDropout(),
        DenseResize(),
        Flux.Dense(16, Nout),
        ParamsScale(), # Scale to parameter space
        @λ(x -> vcat(Flux.softmax(x[1:2,:]), Flux.relu.(x[3:4,:]))),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function ConvResNet(
        info      :: DataInfo{T} = DataInfo();
        Nkern     :: Int = 3,
        Nfeat     :: Int = 4,
        Nconv     :: Int = 4,
        Nblock    :: Int = 4,
        act       :: Symbol = :leakyrelu,
        dropout   :: T = T(0.0),
        batchnorm :: Bool = true,
        groupnorm :: Bool = false,
    ) where {T}

    @assert !(batchnorm && groupnorm)
    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H       = nfeatures # Data height
    C       = nchannels # Number of channels
    Nout    = nlabels # output size
    actfun  = make_activation(act)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Scale(labwidth, labmean)
    MakeDropout() = (dropout > 0) ? Flux.Dropout(dropout) : identity
    Upsample() = XavierConv((1,1), 1 => Nfeat, actfun; pad = (0,0)) # 1x1 upsample convolutions
    Downsample() = XavierConv((1,1), Nfeat => 1, actfun; pad = (0,0)) # 1x1 downsample convolutions
    ResidualBlock() = IdentitySkip(
        BatchConvConnection((Nkern,Nkern), Nfeat => Nfeat, actfun;
        numlayers = Nconv, batchnorm = batchnorm, groupnorm = groupnorm, mode = :post))

    # Residual network
    residualnetwork = Flux.Chain(
        # DenseResize(),
        # Flux.Dense(H*C, H*C ÷ 2, actfun),
        # ChannelResize(1),
        Upsample(),
        (ResidualBlock() for _ in 1:Nblock)...,
        MakeDropout(),
        Downsample(),
        DenseResize(),
        Flux.Dense(H*C, Nout),
    )

    # Output params
    model = Flux.Chain(
        residualnetwork,
        ParamsScale(), # Scale to parameter space
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            # Flux.relu.(x[3:5,:]), # Positive parameters
            # # x[6:6,:], # Unbounded parameters
        )),
    )

    return model |> m -> Flux.paramtype(T, m)
end

"""
    Residual Dense Network for Image Super-Resolution:
        https://arxiv.org/abs/1802.08797
"""
function ResidualDenseNet(
        info      :: DataInfo{T} = DataInfo();
        act       :: Symbol = :relu,      # Activation function
        dropout   :: Bool   = false,      # Use batch normalization
        batchnorm :: Bool   = true,       # Use batch normalization
        groupnorm :: Bool   = false,      # Use group normalization
        batchmode :: Symbol = :pre,       # Batchnorm mode for BatchConvConnection
        factory   :: Symbol = :batchconv, # Factory type for BatchConvConnection
        Nkern     :: Int    = 3,          # Convolution kernel size
        Nconv     :: Int    = 2,          # Convolutions per BatchConvConnection
        Nfeat     :: Int    = 4,          # Number of features to upsample to from 1-feature input
        Nblock    :: Int    = 2,          # Number of blocks in densely connected RDB layer
        Ndense    :: Int    = 2,          # Number of blocks in GlobalFeatureFusion concatenation layer
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert !(batchnorm && groupnorm)
    H, C, θ = nfeatures, nchannels, nlabels
    actfun = make_activation(act)

    MakeActfun() = @λ(x -> actfun.(x))
    ParamsScale() = Scale(labwidth, labmean)
    MakeDropout() = dropout ? Flux.AlphaDropout(T(0.5)) : identity
    Resample(ch) = XavierConv((1,1), ch; pad = (0,0)) # 1x1 resample convolutions
    
    function DFF()
        local G0, G, C, D, k = Nfeat, Nfeat, Nblock, Ndense, (Nkern,1)
        ConvFactory = @λ(ch -> XavierConv(k, ch, actfun; pad = (k.-1).÷2))
        BatchConvFactory = @λ(ch -> BatchConvConnection(k, ch, actfun; numlayers = Nconv, batchnorm = batchnorm, groupnorm = groupnorm, mode = batchmode))
        # Factory = ConvFactory
        Factory = BatchConvFactory
        DenseFeatureFusion(Factory, G0, G, C, D, k, actfun; dims = 3)
    end

    # Residual network
    residualdensenet = Flux.Chain(
        # ChannelResize(4),
        # ChannelwiseDense(H*C ÷ 4, 4 => 1, actfun),
        # DenseResize(),
        # Flux.Dense(H*C, H*C, actfun),
        ChannelResize(C),
        Resample(C => Nfeat),
        DFF(),
        # MakeDropout(),
        batchnorm ? Flux.BatchNorm(Nfeat, actfun) : groupnorm ? Flux.GroupNorm(Nfeat, Nfeat÷2, actfun) : identity,
        Resample(Nfeat => 1),
        DenseResize(),
        # Flux.Dense(H ÷ 4, θ),
        Flux.Dense(H, θ),
    )

    # Output parameter handling:
    #   `relu` to force positivity
    #   `softmax` to force positivity and unit sum, i.e. fractions
    #   `sigmoid` to force positivity for parameters which vary over several orders of magnitude
    model = Flux.Chain(
        residualdensenet,
        ParamsScale(), # Scale to parameter space
        # @λ(x -> Flux.relu.(x)),
        @λ(x -> vcat(
            # Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            # 
            # Flux.relu.(x[1:3,:]), # Positive parameters
            # x[4:4,:], # Unbounded parameters
            # 
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
            # x[6:6,:], # Unbounded parameters
        )),
    )

    return model |> m -> Flux.paramtype(T, m)
end

"""
    RDN modification, removing all residual-like skip connections
"""
function TestModel4(
        info      :: DataInfo{T} = DataInfo();
        act       :: Symbol = :relu,      # Activation function
        dropout   :: Bool   = false,      # Use batch normalization
        batchnorm :: Bool   = true,       # Use batch normalization
        groupnorm :: Bool   = false,      # Use group normalization
        batchmode :: Symbol = :pre,       # Batchnorm mode for BatchConvConnection
        factory   :: Symbol = :batchconv, # Factory type for BatchConvConnection
        Nkern     :: Int    = 3,          # Convolution kernel size
        Nconv     :: Int    = 2,          # Convolutions per BatchConvConnection
        Nfeat     :: Int    = 4,          # Number of features to upsample to from 1-feature input
        Nblock    :: Int    = 2,          # Number of blocks in densely connected RDB layer
        Ndense    :: Int    = 2,          # Number of blocks in GlobalFeatureFusion concatenation layer
        Nglobal   :: Int    = 2,          # Number of GlobalFeatureFusion concatenation layers
        Nstride   :: Int    = 1,          # Number of GlobalFeatureFusion concatenation layers
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert !(batchnorm && groupnorm)
    H, C, θ = nfeatures, nchannels, nlabels
    actfun = make_activation(act)

    SKIP_CONNECTIONS = true
    MAYBESKIP(layer) = SKIP_CONNECTIONS ? IdentitySkip(layer) : layer

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Scale(labwidth, labmean)
    MakeDropout() = dropout ? Flux.AlphaDropout(T(0.5)) : identity
    
    function GFF()
        local G0, G, C, D, k, σ = Nfeat, Nfeat, Nblock, Ndense, (Nkern,1), actfun
        ConvFactory = @λ ch -> XavierConv(k, ch, σ; pad = (k.-1).÷2)
        BatchConvFactory = @λ ch -> BatchConvConnection(k, ch, σ; numlayers = Nconv, batchnorm = batchnorm, groupnorm = groupnorm, mode = batchmode)
        Factory = factory == :conv ? ConvFactory : BatchConvFactory
        gff_layers = []
        for g in 1:Nglobal
            if g > 1
                groupnorm ?
                    push!(gff_layers, Flux.GroupNorm(D * (g - 1) * G0, (D * (g - 1) * G0) ÷ 2, σ)) :
                    push!(gff_layers, Flux.BatchNorm(D * (g - 1) * G0, σ))
                push!(gff_layers, XavierConv((1,1), D * (g - 1) * G0 => g * G0; pad = (0,0), stride = (Nstride,1)))
            end
            push!(gff_layers, GlobalFeatureFusion(3, [MAYBESKIP(DenseConnection(Factory, g * G0, g * G, C; dims = 3)) for d in 1:D]...,))
        end
        groupnorm ?
            push!(gff_layers, Flux.GroupNorm(D * Nglobal * G0, (D * Nglobal * G0) ÷ 2, σ)) :
            push!(gff_layers, Flux.BatchNorm(D * Nglobal * G0, σ))
        push!(gff_layers, XavierConv((1,1), D * Nglobal * G0 => (Nglobal + 1) * G0; pad = (0,0), stride = (Nstride,1)))
        return Flux.Chain(gff_layers...)
    end

    @inline stride_size(N::Int, str::Int, nstr::Int) = (@assert nstr >= 0; nstr == 0 ? N : stride_size(length(1:str:N), str, nstr-1))
    ApplyPool = Nstride > 1
    Hlast = ApplyPool ? 1 : stride_size(H, Nstride, Nglobal)
    Npool = ApplyPool ? stride_size(H, Nstride, Nglobal) - 1 : 1

    # Residual network
    globalfeaturenet = Flux.Chain(
        # ChannelwiseDense(H, C => C),
        # XavierConv((1,1), C => C; pad = (0,0), stride = (2,1)), # spatial downsampling
        XavierConv((Nkern,1), C => Nfeat, actfun; pad = ((Nkern,1).-1).÷2), # channel upsampling
        # XavierConv((Nkern,1), C => Nfeat, actfun; pad = ((Nkern,1).-1).÷2, stride = (2,1)), # spatial downsampling, channel upsampling
        GFF(),
        # XavierConv((Nkern,1), Nglobal * Nfeat => Nglobal * Nfeat, actfun; pad = ((Nkern,1).-1).÷2, stride = (2,1)), # spatial downsampling
        ApplyPool ? Flux.MeanPool((Npool,1)) : identity,
        DenseResize(),
        Flux.Dense(Hlast * (Nglobal + 1) * Nfeat, θ),
    )

    # Output parameter handling:
    #   `relu` to force positivity
    #   `softmax` to force positivity and unit sum, i.e. fractions
    #   `sigmoid` to force positivity for parameters which vary over several orders of magnitude
    model = Flux.Chain(
        globalfeaturenet,
        ParamsScale(), # Scale to parameter space
        # @λ(x -> Flux.relu.(x)),
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:θ,:]), # Positive parameters
        )),
    )

    return model |> m -> Flux.paramtype(T, m)
end

"""
    DeepResNet
"""
function DeepResNet(
        info::DataInfo{T} = DataInfo();
        type::Symbol = :ResNet18, # Type of DeepResNet
        nfilt::Int = 4, # Base number of resnet filters
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    H, C, θ = nfeatures, nchannels, nlabels
    nfilt_last = type ∈ (:ResNet18, :ResNet34) ? nfilt * 8 : nfilt * 32
    ParamsScale() = Scale(labwidth, labmean)

    top_in = Flux.Chain(
        ChannelwiseDense(H, C => C),
        XavierConvTrans(ResNet._rep(7), C => nfilt; pad = ResNet._pad(3), stride = ResNet._rep(4)),
        # XavierConv(ResNet._rep(7), C => nfilt; pad = ResNet._pad(3), stride = ResNet._rep(1)),
        Flux.MaxPool(ResNet._rep(3), pad = ResNet._pad(1), stride = ResNet._rep(2)),
    )

    bottom_in = Flux.Chain(
        # Flux.MeanPool(ResNet._rep(7)),
        Flux.MeanPool(ResNet._rep(5)),
        @λ(x -> reshape(x, :, size(x,4))),
        Flux.Dense(nfilt_last, θ),
        ParamsScale(),
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
        )),
    )

    resnet_maker =
        type == :ResNet18  ? resnet18  : type == :ResNet34  ? resnet34  : type == :ResNet50 ? resnet50 :
        type == :ResNet101 ? resnet101 : type == :ResNet152 ? resnet152 :
        error("Unknown DeepResNet type: " * type)
    resnet = resnet_maker(top_in, bottom_in; initial_filters = nfilt)

    return resnet |> m -> Flux.paramtype(T, m)
end

function BasicHeight32Model1(info::DataInfo{T} = DataInfo()) where {T}
    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert nfeatures == 32
    H, C, θ = nfeatures, nchannels, nlabels

    ParamsScale() = Scale(labwidth, labmean)

    CatConvLayers = let k = 5
        c1 = XavierConv((k,1), C=>8; pad = ((k÷2)*1,0), dilation = 1) # TODO: `DepthwiseConv` broken?
        c2 = XavierConv((k,1), C=>8; pad = ((k÷2)*2,0), dilation = 2) # TODO: `DepthwiseConv` broken?
        c3 = XavierConv((k,1), C=>8; pad = ((k÷2)*3,0), dilation = 3) # TODO: `DepthwiseConv` broken?
        c4 = XavierConv((k,1), C=>8; pad = ((k÷2)*4,0), dilation = 4) # TODO: `DepthwiseConv` broken?
        @λ(x -> cat(c1(x), c2(x), c3(x), c4(x); dims = 3))
    end

    model = Flux.Chain(
        CatConvLayers,
        Flux.BatchNorm(32, Flux.relu),
        XavierConv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # XavierConv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # XavierConv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # XavierConv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.MaxPool((3,1)),
        # Flux.MeanPool((3,1)),
        # Flux.Dropout(0.3),
        DenseResize(),
        Flux.Dense(512, θ),
        ParamsScale(),
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            # Flux.relu.(x[3:θ,:]), # Positive parameters
            x[3:θ,:],
        )),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function BasicHeight32Generator1(info::DataInfo{T} = DataInfo()) where {T}
    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert nfeatures == 32
    @assert nchannels == 1
    H, θ = nfeatures, nlabels

    ParamsScale() = Scale(labwidth, labmean) # Scales approx [-1,1] to param range
    InverseParamsScale() = Scale(inv.(labwidth), -labmean./labwidth) # Scales param range to approx [-1,1]

    EvenSigns = convert(Vector{T}, cospi.(0:31))
    OddSigns  = convert(Vector{T}, cospi.(1:32))
    function Corrections(F = 64, C = 8, k = 3, p = 5, σ = Flux.relu)
        Flux.Chain(
            InverseParamsScale(),
            Flux.Dense(θ, F, σ),
            ChannelResize(1),
            XavierConv((1,1), 1=>C; pad = (0,0)),
            # Flux.Dense(θ, 32*C, σ),
            # ChannelResize(C),
            IdentitySkip(
                Flux.Chain(
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    @λ(x -> σ.(x)),
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C),
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C, σ),
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C),
                ),
            ),
            XavierConv((k,1), 1C=>2C; pad = (k÷2,0)),
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            IdentitySkip(
                Flux.Chain(
                    XavierConv((k,1), 2C=>2C; pad = (k÷2,0)),
                    @λ(x -> σ.(x)),
                    XavierConv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C),
                    # XavierConv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C, σ),
                    # XavierConv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C),
                ),
            ),
            XavierConv((k,1), 2C=>4C; pad = (k÷2,0)),
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            # XavierConv((1,1), C=>1; pad = (0,0)),
            DenseResize(),
            Flux.Dense((F÷16)*4C, 32),
            # @λ(x -> EvenSigns .* x), # Corrections typically alternate sign
        )
    end

    model = Flux.Chain(
        Sumout(
            ForwardProp8Arg(),
            Corrections(),
        ),
        @λ(x -> Flux.relu.(x)),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function BasicHeight32Generator2(info::DataInfo{T} = DataInfo()) where {T}
    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert nfeatures == 32
    @assert nchannels == 1
    H, θ = nfeatures, nlabels

    ParamsScale() = Scale(labwidth, labmean) # Scales approx [-1,1] to param range
    InverseParamsScale() = Scale(inv.(labwidth), -labmean./labwidth) # Scales param range to approx [-1,1]

    OddSigns  = 1:H .|> n -> isodd(n)  ? one(T) : -one(T)
    EvenSigns = 1:H .|> n -> iseven(n) ? one(T) : -one(T)
    function ResRefinement(F = 32, C = 16, R = 4, k = 3, σ = Flux.relu)
        Flux.Chain(
            [IdentitySkip(
                Flux.Chain(
                    ChannelResize(1),
                    XavierConv((1,1), 1=>C; pad = (0,0)), # Upsample channels
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    # @λ(x -> σ.(x)),
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C, σ),
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    XavierConv((1,1), C=>1; pad = (0,0)), # Downsample channels
                    DenseResize(),
                ),
            ) for i in 1:R]...,
        )
    end

    function ResCorrection(F = 32, C = 16, R = 4, k = 3, p = 5, σ = Flux.relu)
        Flux.Chain(
            InverseParamsScale(),
            Flux.Dense(θ, F, σ),
            ChannelResize(1),
            XavierConv((1,1), 1=>C; pad = (0,0)),
            [IdentitySkip(
                Flux.Chain(
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    # @λ(x -> σ.(x)),
                    # XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C, σ),
                    XavierConv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                ),
            ) for i in 1:R]...,
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            DenseResize(),
            Flux.Dense((F÷4)*C, H),
            Scale(OddSigns), # Corrections typically alternate sign
        )
    end

    function DenseCorrection(Fs = (16,16,32,32), σ = Flux.leakyrelu)
        Fs = (θ, Fs...)
        Flux.Chain(
            InverseParamsScale(),
            [Flux.Chain(
                Flux.Dense(Fs[i], Fs[i+1], σ),
                ChannelResize(1),
                Flux.BatchNorm(1),
                DenseResize(),
            ) for i in 1:length(Fs)-1]...,
            Flux.Dense(Fs[end], H, σ),
            Scale(OddSigns), # Corrections typically alternate sign
        )
    end

    function RecurseCorrect(layer)
        return Flux.Chain(
            Sumout(
                layer,
                DenseCorrection(),
            ),
            ResRefinement(),
        )
    end

    model = Flux.Chain(
        ForwardProp8Arg() |> RecurseCorrect,
        # ForwardProp8Arg() |> RecurseCorrect |> RecurseCorrect |> RecurseCorrect |> RecurseCorrect,
        @λ(x -> Flux.relu.(x)),
    )

    return model |> m -> Flux.paramtype(T, m)
end

function BasicDCGAN1(info::DataInfo{T} = DataInfo()) where {T}
    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    @assert nfeatures == 32
    @assert nchannels == 1
    @assert mod(nfeatures, 8) == 0
    H, θ = nfeatures, nlabels

    ParamsScale() = Scale(labwidth, labmean) # Scales approx [-0.5,0.5] to param range
    InverseParamsScale() = Scale(inv.(labwidth), .-labmean./labwidth) # Scales param range to approx [-1,1]

    function Generator()
        C,  k  = 32, (3,1)
        s1, p1 = (1,1), (1,1,0,0)
        s2, p2 = (2,1), (1,0,0,0) # Note: asymmetric padding
        return Sumout(
            ForwardProp8Arg(),
            Flux.Chain(
                InverseParamsScale(),
                Flux.Dense(θ, (H÷8) * C),
                ChannelResize(C),
                XavierConvTrans(k, C => C÷2; pad = p2, stride = s2), # Output height: H÷4
                # Flux.BatchNorm(C÷2),
                @λ(x -> Flux.relu.(x)),
                XavierConvTrans(k, C÷2 => C÷4; pad = p2, stride = s2), # Output height: H÷2
                # Flux.BatchNorm(C÷4),
                @λ(x -> Flux.relu.(x)),
                XavierConvTrans(k, C÷4 => 1; pad = p2, stride = s2), # Output height: H
                DenseResize(),
                Flux.Diagonal(H; initα = (args...) -> T(0.02) .* randn(T, args...), initβ = (args...) -> zeros(T, args...)),
            ),
        )
    end

    function Discriminator()
        C, k, p = 8, (3,1), (1,0)
        s1, s2 = (1,1), (2,1)
        return Flux.Chain(
            ChannelResize(1),
            XavierConv(k, 1 => C; pad = p, stride = s1), # Output height: H
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            XavierConv(k, C => 2C; pad = p, stride = s2), # Output height: H÷2
            # Flux.BatchNorm(2C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            XavierConv(k, 2C => 4C; pad = p, stride = s2), # Output height: H÷4
            # Flux.BatchNorm(4C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            XavierConv(k, 4C => 8C; pad = p, stride = s2), # Output height: H÷8
            # Flux.BatchNorm(8C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            DenseResize(),
            Flux.Dense(8C * (H÷8), 1, Flux.sigmoid),
        )
    end

    sampler = ParamsScale()
    Z = @λ(Nbatch -> sampler(rand(T, θ, Nbatch) .- T(0.5)))
    G = Flux.Chain(Generator(), @λ(x -> Flux.relu.(x))) |> m -> Flux.paramtype(T, m)
    D = Discriminator() |> m -> Flux.paramtype(T, m)

    return Dict("genatr" => G, "discrm" => D, "sampler" => Z)
end

"""
    Bayesian parameter estimation using conditional variational autoencoders for
    gravitational-wave astronomy: https://arxiv.org/abs/1802.08797
        y: input data (height Ny, channels Cy)
        x: true parameters (height Nx)
"""
struct LIGOCVAE{E1,E2,D}
    E1 :: E1
    E2 :: E2
    D  :: D
end
Flux.@functor LIGOCVAE
Base.show(io::IO, m::LIGOCVAE) = model_summary(io, Dict("E1" => m.E1, "E2" => m.E2, "D" => m.D))

Flux.testmode!(m::LIGOCVAE, mode = true) = (map(x -> Flux.testmode!(x, mode), [m.E1, m.E2, m.D]); m)

# Split `μ` into mean and standard deviation
split_mean_std(μ) = (μ[1:end÷2, ..], μ[end÷2+1:end, ..])

# Sample from the diagonal multivariate normal distbn defined by `μ`
sample_mv_normal(μ) = sample_mv_normal(split_mean_std(μ)...)
sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}) where {T} = μ0 .+ σ .* randn(T, size(σ))
sample_mv_normal(μ0::AbstractMatrix{T}, σ::AbstractMatrix{T}, nsamples::Int) where {T} = μ0 .+ σ .* randn(T, size(σ)..., nsamples)
MvNormalSampler() = sample_mv_normal

# Exponentiate logsigma -> sigma
exp_std(μ0, logσ) = vcat(μ0, exp.(logσ))
exp_std(μ) = exp_std(split_mean_std(μ)...)
ExpStd() = exp_std

# Bound mean in (-0.5, 0.5), exponentiate logsigma -> sigma
bound_mu_exp_std(μ0, logσ) = vcat(tanh.(μ0)./2, exp.(logσ))
bound_mu_exp_std(μ) = bound_mu_exp_std(split_mean_std(μ)...)
BoundMuExpStd() = bound_mu_exp_std

# TODO: Tracker was much faster differentiating square.(x) than x.^2 - check for Zygote?
square(x) = x*x

function (m::LIGOCVAE)(y; nsamples::Int = 100, stddev = false)
    @assert nsamples ≥ ifelse(stddev, 2, 1)

    μr0, σr = m.E1(y) |> split_mean_std
    function sample_rθ_posterior()
        zr = sample_mv_normal(μr0, σr)
        μx0, σx = m.D((zr,y)) |> split_mean_std
        return sample_mv_normal(μx0, σx)
    end

    smooth(a, b, γ) = a + γ * (b - a)
    μx = sample_rθ_posterior()
    σ2x = zero(μx)
    μx_last = zero(μx)
    for i in 2:nsamples
        x = sample_rθ_posterior()
        μx_last .= μx
        μx .= smooth.(μx, x, 1//i)
        σ2x .= smooth.(σ2x, (x .- μx) .* (x .- μx_last), 1//i)
    end

    return stddev ? vcat(μx, sqrt.(σ2x)) : μx
end

# Cross-entropy loss function
function H_LIGOCVAE(m::LIGOCVAE, x::AbstractArray{T}, y::AbstractArray{T}; gamma::T = one(T)) where {T}
    μr0, σr = split_mean_std(m.E1(y))
    μq0, σq = split_mean_std(m.E2((x,y)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_std(m.D((zq,y)))

    Zdim, Xout, Nbatch = size(zq,1), size(μx0,1), size(x,2)
    σr2, σq2 = square.(σr), square.(σq)
    σx2, xout = square.(σx), x[1:Xout,:]

    KLdiv = sum(@. (σq2 + square(μr0 - μq0)) / σr2 + log(σr2 / σq2)) / (2*Nbatch) # KL-divergence contribution to cross-entropy (Note: dropped constant -Zdim/2 term)
    ELBO = sum(@. square(xout - μx0) / σx2 + log(σx2)) / (2*Nbatch) # Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +Zdim*log(2π)/2 term)

    return gamma * ELBO + KLdiv
end

# KL-divergence contribution to cross-entropy
function KL_LIGOCVAE(m::LIGOCVAE, x::AbstractArray{T}, y::AbstractArray{T}) where {T}
    μr0, σr = split_mean_std(m.E1(y))
    μq0, σq = split_mean_std(m.E2((x,y)))

    Zdim, Nbatch = size(μq0,1), size(x,2)
    σr2, σq2 = square.(σr), square.(σq)

    KLdiv = sum(@. (σq2 + square(μr0 - μq0)) / σr2 + log(σr2 / σq2)) / (2*Nbatch) # KL-divergence contribution to cross-entropy (Note: dropped constant -Zdim/2 term)

    return KLdiv
end

# Negative log-likelihood/ELBO loss function
function L_LIGOCVAE(m::LIGOCVAE, x::AbstractArray{T}, y::AbstractArray{T}) where {T}
    μq0, σq = split_mean_std(m.E2((x,y)))
    zq = sample_mv_normal(μq0, σq)
    μx0, σx = split_mean_std(m.D((zq,y)))

    Zdim, Xout, Nbatch = size(zq,1), size(μx0,1), size(x,2)
    σx2, xout = square.(σx), x[1:Xout,:]

    ELBO = sum(@. square(xout - μx0) / σx2 + log(σx2)) / (2*Nbatch) # Negative log-likelihood/ELBO contribution to cross-entropy (Note: dropped constant +Zdim*log(2π)/2 term)

    return ELBO
end

function DenseLIGOCVAE(
        info      :: DataInfo{T} = DataInfo();
        Xout      :: Int = info.nlabels, # Number of output variables (can be used to marginalize over inputs)
        Zdim      :: Int = 10, # Latent variable dimensions
        Nh        :: Int = 2, # Number of inner hidden dense layers
        Dh        :: Int = info.nfeatures, # Dimension of inner hidden dense layers
        dropout   :: T = T(0.0), # Dropout following dense layer (0.0 is none)
        boundmean :: Bool = false, # Bound mean to be within labmean +/- labwidth/2
        act       :: Symbol = :leakyrelu, # Activation function
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    Ny, Cy, Nx = nfeatures, nchannels, nlabels
    actfun = make_activation(act)

    MuStdScale() = Scale(T[labwidth[1:Xout]; labwidth[1:Xout]], T[labmean[1:Xout]; zeros(T, Xout)]) # output μ_x = [μ_x0; σ_x] vector -> μ_x0: scaled and shifted ([-0.5, 0.5] -> x range), σ_x: scaled only
    XYScale() = Scale([inv.(labwidth); ones(T, Ny*Cy)], [-labmean./labwidth; zeros(T, Ny*Cy)]) # [x; y] vector -> x scaled and shifted (x range -> [-0.5, 0.5]), y unchanged
    MaybeDropout() = dropout == 0 ? identity : Flux.Dropout(dropout)

    # MLP layers (with no activation on output)
    function MLPlayers(in, out, act)
        l = Any[]
        push!(l, Flux.Dense(in, Dh, act))
        (dropout > 0) && push!(l, Flux.Dropout(dropout))
        for _ in 1:Nh
            push!(l, Flux.Dense(Dh, Dh, act))
            (dropout > 0) && push!(l, Flux.Dropout(dropout))
        end
        push!(l, Flux.Dense(Dh, out))
        return l
    end

    # Data/feature encoder r_θ1(z|y): y -> μ_r = [μ_r0; σ_r]
    E1 = Flux.Chain(
        DenseResize(),
        MLPlayers(Ny*Cy, 2*Zdim, actfun)...,
        boundmean ? BoundMuExpStd() : ExpStd(),
        # ExpStd()
    )

    # Data/feature + parameter/label encoder q_φ(z|x,y): (x,y) -> μ_q = [μ_q0; σ_q]
    E2 = Flux.Chain(
        @λ(((x,y),) -> vcat(x, DenseResize()(y))),
        XYScale(),
        MLPlayers(Nx + Ny*Cy, 2*Zdim, actfun)...,
        boundmean ? BoundMuExpStd() : ExpStd(),
        # ExpStd(),
    )

    # Latent space + data/feature decoder r_θ2(x|z,y): (z,y) -> μ_x = [μ_x0; σ_x]
    D = Flux.Chain(
        @λ(((z,y),) -> vcat(z, DenseResize()(y))),
        MLPlayers(Zdim + Ny*Cy, 2*Xout, actfun)...,
        boundmean ? BoundMuExpStd() : ExpStd(),
        MuStdScale(),
    )

    return LIGOCVAE(E1, E2, D) |> m -> Flux.paramtype(T, m) # convert internal float types to T
end

function ConvLIGOCVAE(
        info   :: DataInfo{T} = DataInfo();
        Xout   :: Int = info.nlabels, # Number of output variables (can be used to marginalize over inputs)
        Zdim   :: Int = 10, # Latent variable dimensions
        act    :: Symbol = :leakyrelu, # Activation function
        Nfeat  :: Int = 8, # Number of convolutional channels
        Ndown  :: Int = 1, # Optional striding for downsampling
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    Ny, Cy, Nx = nfeatures, nchannels, nlabels
    actfun = make_activation(act)

    MuStdScale() = Scale(T[labwidth[1:Xout]; labwidth[1:Xout]], T[labmean[1:Xout]; zeros(T, Xout)]) # output μ_x = [μ_x0; σ_x] vector -> μ_x0: scaled and shifted ([-0.5, 0.5] -> x range), σ_x: scaled only
    XYScale() = Scale([inv.(labwidth); ones(T, Ny*Cy)], [-labmean./labwidth; zeros(T, Ny*Cy)]) # [x; y] vector -> x scaled and shifted (x range -> [-0.5, 0.5]), y unchanged
    XScale() = Scale(inv.(labwidth), -labmean./labwidth) # x scaled and shifted (x range -> [-0.5, 0.5])

    PreprocessSignal(k = (3,1), ch = 1=>8, N = 2, BN = true, GN = false) = Flux.Chain(
        XavierConv((1,1), ch[1] => ch[2]; pad = (0,0)), # Channel upsample
        IdentitySkip( # Residual connection
            BatchConvConnection(k, ch[2] => ch[2], actfun; mode = :pre, numlayers = N, batchnorm = BN, groupnorm = GN),
        ),
        XavierConv(k, ch[2] => ch[2]; pad = (k.-1).÷2, stride = Ndown), # Spatial downsampling
        IdentitySkip( # Residual connection
            BatchConvConnection(k, ch[2] => ch[2], actfun; mode = :pre, numlayers = N, batchnorm = BN, groupnorm = GN),
        ),
        XavierConv(k, ch[2] => ch[2]; pad = (k.-1).÷2, stride = Ndown), # Spatial downsampling
        DenseResize(),
    )

    # Data/feature encoder r_θ1(z|y): y -> μ_r = (μ_r0, σ_r^2)
    E1 = let Nout = 2*Zdim, k = (3,1), ch = Cy=>Nfeat
        Nd = [ch[2]*Ny÷Ndown^2, Ny, Nout]
        AF(i) = i == length(Nd)-1 ? identity : actfun
        Flux.Chain(
            PreprocessSignal(k, ch),
            DenseResize(),
            [Flux.Dense(Nd[i], Nd[i+1], AF(i)) for i in 1:length(Nd)-1]...,
            ExpStd(),
        )
    end

    # Data/feature + parameter/label encoder q_φ(z|x,y): (x,y) -> μ_q = (μ_q0, σ_q^2)
    E2 = let Nout = 2*Zdim, k = (3,1), ch = Cy=>Nfeat
        Nd = [Nx + ch[2]*Ny÷Ndown^2, Ny, Nout]
        AF(i) = i == length(Nd)-1 ? identity : actfun
        Flux.Chain(
            MultiInput(
                XScale(), # x parameters scaled down to roughly [-0.5, 0.5]
                PreprocessSignal(k, ch), # y signals preprocessed
            ),
            @λ(((x,y),) -> vcat(x, y)),
            [Flux.Dense(Nd[i], Nd[i+1], AF(i)) for i in 1:length(Nd)-1]...,
            ExpStd(),
        )
    end

    # Latent space + data/feature decoder r_θ2(x|z,y): (z,y) -> μ_x = (μ_x0, σ_x^2)
    D = let Nout = 2*Xout, k = (3,1), ch = Cy=>Nfeat
        Nd = [Zdim + ch[2]*Ny÷Ndown^2, Ny, Nout]
        AF(i) = i == length(Nd)-1 ? identity : actfun
        Flux.Chain(
            MultiInput(
                identity, # z vector of latent variables unchanged
                PreprocessSignal(k, ch), # y signals preprocessed
            ),
            @λ(((z,y),) -> vcat(z, y)),
            [Flux.Dense(Nd[i], Nd[i+1], AF(i)) for i in 1:length(Nd)-1]...,
            ExpStd(),
            MuStdScale(), # scale and shift μ_x = [μ_x0; σ_x] to x-space
        )
    end

    return LIGOCVAE(E1, E2, D) |> m -> Flux.paramtype(T, m) # convert internal float types to T
end

function RDNLIGOCVAE(
        info :: DataInfo{T} = DataInfo();
        Xout :: Int = info.nlabels, # Number of output variables (can be used to marginalize over inputs)
        Zdim :: Int = 10, # Latent variable dimensions
        Nrdn :: Int = 2, # Number of RDN + downsampling blocks
        Ncat :: Int = 2, # Number of concat blocks within RDN
        Nfeat:: Int = 8, # Number of convolutional channels + RDN growth rate
        act  :: Symbol = :leakyrelu, # Activation function
    ) where {T}

    @unpack nfeatures, nchannels, nlabels, labmean, labwidth = info
    Ny, Cy, Nx = nfeatures, nchannels, nlabels
    actfun = make_activation(act)

    MuStdScale() = Scale(T[labwidth[1:Xout]; labwidth[1:Xout]], T[labmean[1:Xout]; zeros(T, Xout)]) # output μ_x = [μ_x0; σ_x] vector -> μ_x0: scaled and shifted ([-0.5, 0.5] -> x range), σ_x: scaled only
    XScale() = Scale(inv.(labwidth), -labmean./labwidth) # x scaled and shifted (x range -> [-0.5, 0.5])
    Dσ(i,ND) = i == ND-1 ? identity : actfun # Activation functions for ND dense layers

    PreprocessSignal(k = (3,1), ch = 1=>8) = Flux.Chain(
        XavierConv(k, ch[1] => ch[2]; pad = (k.-1).÷2), # Expand channels
        mapreduce(vcat, 1:Nrdn; init = []) do _
            [
                ResidualDenseBlock(ch[2], ch[2], Ncat; dims = 3, k = k, σ = actfun), # RDB block
                Flux.BatchNorm(ch[2], actfun), # Batchnorm + pre-activation
                XavierConv(k, ch[2] => ch[2]; pad = (k.-1).÷2, stride = 2), # Spatial downsampling
            ]
        end...,
        DenseResize(),
    )

    # Data/feature encoder r_θ1(z|y): y -> μ_r = (μ_r0, σ_r^2)
    E1 = let Nout = 2*Zdim, k = (3,1)
        Dy  = [Nfeat*Ny÷(2^Nrdn), Ny, Nout]
        NDy = length(Dy)
        Flux.Chain(
            PreprocessSignal(k, Cy=>Nfeat),
            DenseResize(),
            [Flux.Dense(Dy[i], Dy[i+1], Dσ(i,NDy)) for i in 1:NDy-1]...,
            ExpStd(),
        )
    end

    # Data/feature + parameter/label encoder q_φ(z|x,y): (x,y) -> μ_q = (μ_q0, σ_q^2)
    E2 = let Nout = 2*Zdim, k = (3,1)
        Dx  = [Nx, 2*Nx, 2*Nx, Nx]
        Dxy = [Dx[end] + Nfeat*Ny÷(2^Nrdn), Ny, Nout]
        NDx, NDxy = length(Dx), length(Dxy)
        Flux.Chain(
            MultiInput(
                Flux.Chain(
                    XScale(), # x parameters scaled down to roughly [-0.5, 0.5]
                    [Flux.Dense(Dx[i], Dx[i+1], Dσ(i,NDx)) for i in 1:NDx-1]...,
                ),
                PreprocessSignal(k, Cy=>Nfeat), # y signals preprocessed
            ),
            @λ(((x,y),) -> vcat(x, y)),
            [Flux.Dense(Dxy[i], Dxy[i+1], Dσ(i,NDxy)) for i in 1:NDxy-1]...,
            ExpStd(),
        )
    end

    # Latent space + data/feature decoder r_θ2(x|z,y): (z,y) -> μ_x = (μ_x0, σ_x^2)
    D = let Nout = 2*Xout, k = (3,1)
        Dyz  = [Zdim + Nfeat*Ny÷(2^Nrdn), Ny, Nout]
        NDyz = length(Dyz)
        Flux.Chain(
            MultiInput(
                identity, # z vector of latent variables unchanged
                PreprocessSignal(k, Cy=>Nfeat), # y signals preprocessed
            ),
            @λ(((z,y),) -> vcat(z, y)),
            [Flux.Dense(Dyz[i], Dyz[i+1], Dσ(i,NDyz)) for i in 1:NDyz-1]...,
            ExpStd(),
            MuStdScale(), # scale and shift μ_x = [μ_x0; σ_x] to x-space
        )
    end

    return LIGOCVAE(E1, E2, D) |> m -> Flux.paramtype(T, m) # convert internal float types to T
end
