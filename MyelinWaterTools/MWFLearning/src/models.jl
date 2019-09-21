const model_dict = Dict(
    "ConvResNet"              => conv_resnet,
    "ResidualDenseNet"        => residual_dense_net,
    "Keras1DSeqClass"         => keras_1D_sequence_classification,
    "TestModel1"              => test_model_1,
    "TestModel2"              => test_model_2,
    "TestModel3"              => test_model_3,
    "TestModel4"              => test_model_4,
    "ResNet"                  => resnet,
    "BasicHeight32Model1"     => basic_height32_model_1,
    "BasicHeight32Generator1" => basic_height32_generator_1,
    "BasicHeight32Generator2" => basic_height32_generator_2,
    "BasicDCGAN1"             => basic_DCGAN_1,
)
function get_model(settings::Dict)
    T = settings["prec"] == 64 ? Float64 : Float32
    models = []
    for (name, model) in settings["model"]
        if name ∈ keys(model_dict)
            kwargs = make_model_kwargs(T, model)
            push!(models, model_dict[name](T; kwargs...))
        end
    end
    models    
end

make_model_kwargs(::Type{T}, m::Dict) where {T} = Dict(Symbol.(keys(m)) .=> clean_model_kwargs.(T, values(m)))
clean_model_kwargs(::Type{T}, x) where {T} = error("Unsupported model parameter $x") # fallback
clean_model_kwargs(::Type{T}, x::Bool) where {T} = x
clean_model_kwargs(::Type{T}, x::Integer) where {T} = Int(x)
clean_model_kwargs(::Type{T}, x::AbstractString) where {T} = Symbol(x)
clean_model_kwargs(::Type{T}, x::AbstractFloat) where {T} = T(x)
clean_model_kwargs(::Type{T}, x::Vector{Tx}) where {T, Tx} = clean_model_kwargs.(T, x)

const activation_dict = Dict(
    "relu"      => NNlib.relu,
    "sigma"     => NNlib.σ,
    "leakyrelu" => NNlib.leakyrelu,
    "elu"       => NNlib.elu,
    "swish"     => NNlib.swish,
    "softplus"  => NNlib.softplus,
)
get_activation(str::String) = activation_dict[str]

"""
See "Sequence classification with 1D convolutions" at the following url:
    https://keras.io/getting-started/sequential-model-guide/
"""
function keras_1D_sequence_classification(settings)
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Npool, Nkern, Nout, act = settings["model"]
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf2 * (H ÷ Npool ÷ Npool)
    actfun = get_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling
        Flux.Conv((Nkern,1), C => Nf1, pad = (Npad,0), actfun), # (H, 1, 1) -> (H, Nf1, 1)
        Flux.Conv((Nkern,1), Nf1 => Nf1, pad = (Npad,0), actfun), # (H, Nf1, 1) -> (H, Nf1, 1)
        Flux.MaxPool((Npool,)), # (H, Nf1, 1) -> (H/Npool, Nf1, 1)

        # Two more convolution layers followed by mean pooling
        Flux.Conv((Nkern,1), Nf1 => Nf2, pad = (Npad,0), actfun), # (H/Npool, Nf1, 1) -> (H/Npool, Nf2, 1)
        Flux.Conv((Nkern,1), Nf2 => Nf2, pad = (Npad,0), actfun), # (H/Npool, Nf2, 1) -> (H/Npool, Nf2, 1)
        Flux.MeanPool((Npool,)), # (H/Npool, Nf2, 1) -> (H/Npool^2, Nf2, 1)

        # Dropout layer
        settings["model"]["dropout"] ? Flux.Dropout(0.5) : identity,

        # Dense + softmax layer
        DenseResize(),
        Flux.Dense(Ndense, Nout, actfun),
        settings["model"]["softmax"] ? NNlib.softmax : identity,
    )
    
    return model
end

function test_model_1(settings)
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Nf3, Npool, Nkern, Nout, act = settings["model"]
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf3 * (H ÷ Npool ÷ Npool ÷ Npool)
    actfun = get_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,1), C => Nf1, pad = (Npad,0), actfun),
        Flux.Conv((Nkern,1), Nf1 => Nf1, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Nf1, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,1), Nf1 => Nf2, pad = (Npad,0), actfun),
        Flux.Conv((Nkern,1), Nf2 => Nf2, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Nf2, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,1), Nf2 => Nf3, pad = (Npad,0), actfun),
        Flux.Conv((Nkern,1), Nf3 => Nf3, pad = (Npad,0), actfun),
        Flux.MaxPool((Npool,)),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Nf3, actfun) : identity,

        # Dropout layer
        settings["model"]["dropout"] ? Flux.Dropout(0.5) : identity,

        # Dense / batchnorm layer
        DenseResize(),
        Flux.Dense(Ndense, Ndense ÷ 2, actfun),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Ndense ÷ 2, actfun) : identity,

        # Dense / batchnorm layer, but last actfun must be softplus since outputs are positive
        settings["model"]["batchnorm"] ? Flux.Dense(Ndense ÷ 2, Nout, actfun) : Flux.Dense(Ndense ÷ 2, Nout, NNlib.softplus),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Nout, NNlib.softplus) : identity,

        # Softmax
        settings["model"]["softmax"] ? NNlib.softmax : identity,

        # Scale from (0,1) back to model parameter range
        settings["model"]["scale"] == false ? identity : Scale(settings["model"]["scale"]),
    )

    return model
end

function test_model_2(settings)
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nout, act, Nd = settings["model"]
    actfun = get_activation(act)

    model = Flux.Chain(
        DenseResize(),
        Flux.Dense(H * C, Nd[1], actfun),
        # settings["model"]["batchnorm"] ? Flux.BatchNorm(Nd[1], actfun) : identity,
        
        reduce(vcat, [
            Flux.Dense(Nd[i], Nd[i+1], actfun),
            # settings["model"]["batchnorm"] ? Flux.BatchNorm(Nd[i+1], actfun) : identity
        ] for i in 1:length(Nd)-1)...,

        Flux.Dense(Nd[end], Nout, actfun),
        settings["model"]["batchnorm"] ? Flux.BatchNorm(Nout, actfun) : identity,

        # Softmax
        settings["model"]["softmax"] ? NNlib.softmax : identity,
        
        # Softplus to ensure positivity, unless softmax has already been applied
        settings["model"]["softmax"] ? identity : @λ(x -> NNlib.softplus.(x)),

        # Scale from (0,1) back to model parameter range
        settings["model"]["scale"] == false ? identity : Scale(settings["model"]["scale"]),
    )

    return model
end

function test_model_3(settings)
    H      = settings["data"]["height"] :: Int # Data height
    C      = settings["data"]["channels"] :: Int # Number of channels
    actfun = settings["model"]["act"] |> get_activation
    Nout   = settings["model"]["Nout"] :: Int # Number of outputs
    BN     = settings["model"]["batchnorm"] :: Bool # Use batch normalization
    scale  = settings["model"]["scale"] :: Vector # Parameter scales
    NP     = settings["data"]["preprocess"]["wavelet"]["apply"] == true ? length(scale) : 0

    MakeActfun() = @λ x -> actfun.(x)
    PhysicalParams() = @λ(x -> x[1:NP, 1:C, :])
    NonPhysicsCoeffs() = @λ(x -> x[NP+1:end, 1:C, :])
    ParamsScale() = settings["model"]["scale"] == false ? identity : Scale(settings["model"]["scale"])
    MakeDropout() = settings["model"]["dropout"] == true ? Flux.AlphaDropout(0.5) : identity
    ResidualBlock() = IdentitySkip(BatchDenseConnection(4, 4, actfun; groupnorm = BN, mode = :post))

    # NonPhysicsCoeffs -> Output Parameters
    residualnetwork = Flux.Chain(
        NonPhysicsCoeffs(),
        DenseResize(),
        Flux.Dense(H*C - NP, 16, actfun),
        ChannelResize(4),
        (ResidualBlock() for _ in 1:3)...,
        MakeDropout(),
        DenseResize(),
        Flux.Dense(16, Nout),
        ParamsScale(), # Scale to parameter space
        NP > 0 ?
            @λ(x -> 0.01 * x) : # Force physics perturbation to be small (helps with convergence)
            Flux.Diagonal(NP) # Learn output scale
    )

    # Reshape physical parameters
    physicsnetwork = Flux.Chain(
        PhysicalParams(),
        DenseResize(),
    )

    # Output (assumes 1:2 is mwf/iewf and 3:4 are other physical params)
    model = Flux.Chain(
        NP > 0 ? Sumout(physicsnetwork, residualnetwork) : residualnetwork,
        @λ(x -> vcat(Flux.softmax(x[1:2,:]), Flux.relu.(x[3:4,:]))),
    )

    return model
end

function conv_resnet(settings)
    H       = settings["data"]["height"] :: Int # Data height
    C       = settings["data"]["channels"] :: Int # Number of channels
    Nout    = settings["model"]["Nout"] :: Int # Number of outputs
    actfun  = settings["model"]["act"] |> get_activation # Activation function
    scale   = settings["model"]["scale"] :: Vector # Parameter scales
    offset  = settings["model"]["offset"] :: Vector # Parameter offsets
    DP      = settings["model"]["densenet"]["dropout"] :: Bool # Use batch normalization
    BN      = settings["model"]["densenet"]["batchnorm"] :: Bool # Use batch normalization
    GN      = settings["model"]["densenet"]["groupnorm"] :: Bool # Use group normalization
    Nkern   = settings["model"]["densenet"]["Nkern"] :: Int # Kernel size
    Nfeat   = settings["model"]["densenet"]["Nfeat"] :: Int # Kernel size
    Nconv   = settings["model"]["densenet"]["Nconv"] :: Int # Num residual connection layers
    Nblock  = settings["model"]["densenet"]["Nblock"] :: Int # Num residual connection layers
    @assert !(BN && GN)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)
    MakeDropout() = DP ? Flux.AlphaDropout(0.5) : identity
    Upsample() = Flux.Conv((1,1), 1 => Nfeat, actfun; init = xavier_uniform, pad = (0,0)) # 1x1 upsample convolutions
    Downsample() = Flux.Conv((1,1), Nfeat => 1, actfun; init = xavier_uniform, pad = (0,0)) # 1x1 downsample convolutions
    ResidualBlock() = IdentitySkip(
        BatchConvConnection(Nkern, Nfeat => Nfeat, actfun;
        numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = :post))

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
        # ParamsScale(), # Scale to parameter space
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            # Flux.relu.(x[3:5,:]), # Positive parameters
            # # x[6:6,:], # Unbounded parameters
        )),
    )

    return model
end

"""
Residual Dense Network for Image Super-Resolution: https://arxiv.org/abs/1802.08797
"""
function residual_dense_net(settings)
    T = settings["prec"] == 64 ? Float64 : Float32
    VT = Vector{T}

    H      :: Int      = settings["data"]["height"] # Data height
    C      :: Int      = settings["data"]["channels"] # Number of channels
    Nout   :: Int      = settings["model"]["Nout"] # Number of outputs
    actfun :: Function = settings["model"]["act"] |> get_activation # Activation function
    scale  :: VT       = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset :: VT       = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    DP     :: Bool     = settings["model"]["densenet"]["dropout"] # Use batch normalization
    BN     :: Bool     = settings["model"]["densenet"]["batchnorm"] # Use batch normalization
    GN     :: Bool     = settings["model"]["densenet"]["groupnorm"] # Use group normalization
    mode   :: Symbol   = settings["model"]["densenet"]["batchmode"] :: String |> Symbol # Batchnorm mode for BatchConvConnection
    Nkern  :: Int      = settings["model"]["densenet"]["Nkern"] # Convolution kernel size
    Nconv  :: Int      = settings["model"]["densenet"]["Nconv"] # Convolutions per BatchConvConnection
    Nfeat  :: Int      = settings["model"]["densenet"]["Nfeat"] # Number of features to upsample to from 1-feature input
    Nblock :: Int      = settings["model"]["densenet"]["Nblock"] # Number of blocks in densely connected RDB layer
    Ndense :: Int      = settings["model"]["densenet"]["Ndense"] # Number of blocks in GlobalFeatureFusion concatenation layer
    @assert !(BN && GN)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)
    MakeDropout() = DP ? Flux.AlphaDropout(T(0.5)) : identity
    Resample(ch) = Flux.Conv((1,1), ch, identity; init = xavier_uniform, pad = (0,0)) # 1x1 resample convolutions
    
    function DFF()
        local G0, G, C, D, k = Nfeat, Nfeat, Nblock, Ndense, (Nkern,1)
        ConvFactory = @λ ch -> Flux.Conv(k, ch, actfun; init = xavier_uniform, pad = (k.-1).÷2)
        BatchConvFactory = @λ ch -> BatchConvConnection(k, ch, actfun; numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = mode)
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
        BN ? Flux.BatchNorm(Nfeat, actfun) : GN ? Flux.GroupNorm(Nfeat, Nfeat÷2, actfun) : identity,
        Resample(Nfeat => 1),
        DenseResize(),
        # Flux.Dense(H ÷ 4, Nout),
        Flux.Dense(H, Nout),
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

    return model
end

"""
Modification of the Residual Dense Network for Image Super-Resolution (https://arxiv.org/abs/1802.08797),
removing all residual-like skip connections.
"""
function test_model_4(settings)
    T = settings["prec"] == 64 ? Float64 : Float32
    VT = Vector{T}

    H       :: Int      = settings["data"]["height"] # Data height
    C       :: Int      = settings["data"]["channels"] # Number of channels
    Nout    :: Int      = settings["model"]["Nout"] # Number of outputs
    actfun  :: Function = settings["model"]["act"] |> get_activation # Activation function
    scale   :: VT       = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset  :: VT       = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    DP      :: Bool     = settings["model"]["densenet"]["dropout"] # Use batch normalization
    BN      :: Bool     = settings["model"]["densenet"]["batchnorm"] # Use batch normalization
    GN      :: Bool     = settings["model"]["densenet"]["groupnorm"] # Use group normalization
    mode    :: Symbol   = settings["model"]["densenet"]["batchmode"] :: String |> Symbol # Batchnorm mode for BatchConvConnection
    factory :: Symbol   = settings["model"]["densenet"]["factory"] :: String |> Symbol # Factory type for BatchConvConnection
    Nkern   :: Int      = settings["model"]["densenet"]["Nkern"] # Convolution kernel size
    Nconv   :: Int      = settings["model"]["densenet"]["Nconv"] # Convolutions per BatchConvConnection
    Nfeat   :: Int      = settings["model"]["densenet"]["Nfeat"] # Number of features to upsample to from 1-feature input
    Nblock  :: Int      = settings["model"]["densenet"]["Nblock"] # Number of blocks in densely connected RDB layer
    Ndense  :: Int      = settings["model"]["densenet"]["Ndense"] # Number of blocks in GlobalFeatureFusion concatenation layer
    Nglobal :: Int      = settings["model"]["densenet"]["Nglobal"] # Number of GlobalFeatureFusion concatenation layers
    Nstride :: Int      = settings["model"]["densenet"]["Nstride"] # Number of GlobalFeatureFusion concatenation layers
    @assert !(BN && GN)

    SKIP_CONNECTIONS = true
    MAYBESKIP(layer) = SKIP_CONNECTIONS ? IdentitySkip(layer) : layer

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)
    MakeDropout() = DP ? Flux.AlphaDropout(T(0.5)) : identity
    
    function GFF()
        local G0, G, C, D, k, σ = Nfeat, Nfeat, Nblock, Ndense, (Nkern,1), actfun
        ConvFactory = @λ ch -> Flux.Conv(k, ch, σ; init = xavier_uniform, pad = (k.-1).÷2)
        BatchConvFactory = @λ ch -> BatchConvConnection(k, ch, σ; numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = mode)
        Factory = factory == :conv ? ConvFactory : BatchConvFactory
        gff_layers = []
        for g in 1:Nglobal
            if g > 1
                GN ? push!(gff_layers, Flux.GroupNorm(D * (g - 1) * G0, (D * (g - 1) * G0) ÷ 2, σ)) :
                     push!(gff_layers, Flux.BatchNorm(D * (g - 1) * G0, σ))
                push!(gff_layers, Flux.Conv((1,1), D * (g - 1) * G0 => g * G0, identity; init = xavier_uniform, pad = (0,0), stride = (Nstride,1)))
            end
            push!(gff_layers, GlobalFeatureFusion(3, [MAYBESKIP(DenseConnection(Factory, g * G0, g * G, C; dims = 3)) for d in 1:D]...,))
        end
        GN ? push!(gff_layers, Flux.GroupNorm(D * Nglobal * G0, (D * Nglobal * G0) ÷ 2, σ)) :
             push!(gff_layers, Flux.BatchNorm(D * Nglobal * G0, σ))
        push!(gff_layers, Flux.Conv((1,1), D * Nglobal * G0 => (Nglobal + 1) * G0, identity; init = xavier_uniform, pad = (0,0), stride = (Nstride,1)))
        return Flux.Chain(gff_layers...)
    end

    @inline stride_size(N::Int, str::Int, nstr::Int) = (@assert nstr >= 0; nstr == 0 ? N : stride_size(length(1:str:N), str, nstr-1))
    ApplyPool = Nstride > 1
    Hlast = ApplyPool ? 1 : stride_size(H, Nstride, Nglobal)
    Npool = ApplyPool ? stride_size(H, Nstride, Nglobal) - 1 : 1

    # Residual network
    globalfeaturenet = Flux.Chain(
        # ChannelwiseDense(H, C => C),
        # Flux.Conv((1,1), C => C, identity; init = xavier_uniform, pad = (0,0), stride = (2,1)), # spatial downsampling
        Flux.Conv((Nkern,1), C => Nfeat, actfun; init = xavier_uniform, pad = ((Nkern,1).-1).÷2), # channel upsampling
        # Flux.Conv((Nkern,1), C => Nfeat, actfun; init = xavier_uniform, pad = ((Nkern,1).-1).÷2, stride = (2,1)), # spatial downsampling, channel upsampling
        GFF(),
        # Flux.Conv((Nkern,1), Nglobal * Nfeat => Nglobal * Nfeat, actfun; init = xavier_uniform, pad = ((Nkern,1).-1).÷2, stride = (2,1)), # spatial downsampling
        ApplyPool ? Flux.MeanPool((Npool,1)) : identity,
        DenseResize(),
        Flux.Dense(Hlast * (Nglobal + 1) * Nfeat, Nout),
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
            Flux.relu.(x[3:5,:]), # Positive parameters
        )),
    )

    return model
end

"""
    ResNet
"""
function resnet(settings)
    VT               = settings["prec"] == 64 ? Vector{Float64} : Vector{Float32}
    H      :: Int    = settings["data"]["height"] :: Int # Data height
    C      :: Int    = settings["data"]["channels"] :: Int # Number of channels
    Nout   :: Int    = settings["model"]["Nout"] :: Int # Number of outputs
    type   :: Symbol = settings["model"]["resnet"]["type"] :: String |> Symbol # Factory type for BatchConvConnection
    Nfilt  :: Int    = settings["model"]["resnet"]["Nfilt"] :: Int # Number of features to upsample to from 1-feature input
    scale  :: VT     = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset :: VT     = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    
    NfiltLast = type ∈ (:ResNet18, :ResNet34) ? Nfilt * 8 : Nfilt * 32
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)

    top_in = Flux.Chain(
        ChannelwiseDense(H, C => C),
        Flux.ConvTranspose(ResNet._rep(7), C => Nfilt; pad = ResNet._pad(3), stride = ResNet._rep(4)),
        # Flux.Conv(ResNet._rep(7), C => Nfilt; pad = ResNet._pad(3), stride = ResNet._rep(1)),
        Flux.MaxPool(ResNet._rep(3), pad = ResNet._pad(1), stride = ResNet._rep(2)),
    )

    bottom_in = Flux.Chain(
        # Flux.MeanPool(ResNet._rep(7)),
        Flux.MeanPool(ResNet._rep(5)),
        @λ(x -> reshape(x, :, size(x,4))),
        Flux.Dense(NfiltLast, Nout),
        ParamsScale(),
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
        )),
    )

    resnetX =
        type == :ResNet18 ? resnet18 : type == :ResNet34  ? resnet34  :
        type == :ResNet50 ? resnet50 : type == :ResNet101 ? resnet101 : type == :ResNet152 ? resnet152 :
        error("Unknown ResNet type: $type")

    resnet = resnetX(top_in, bottom_in; initial_filters = Nfilt)

    return resnet
end

function basic_height32_model_1(settings)
    VT     :: Type   = settings["prec"] == 64 ? Vector{Float64} : Vector{Float32}
    H      :: Int    = settings["data"]["height"] :: Int # Data height
    C      :: Int    = settings["data"]["channels"] :: Int # Number of channels
    Nout   :: Int    = settings["model"]["Nout"] :: Int # Number of outputs
    type   :: Symbol = settings["model"]["resnet"]["type"] :: String |> Symbol # Factory type for BatchConvConnection
    Nfilt  :: Int    = settings["model"]["resnet"]["Nfilt"] :: Int # Number of features to upsample to from 1-feature input
    scale  :: VT     = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset :: VT     = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    
    @assert H == 32 # Model height should be 32

    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)

    CatConvLayers = let k = 5
        c1 = Flux.DepthwiseConv((k,1), 2=>8; pad = ((k÷2)*1,0), dilation = 1)
        c2 = Flux.DepthwiseConv((k,1), 2=>8; pad = ((k÷2)*2,0), dilation = 2)
        c3 = Flux.DepthwiseConv((k,1), 2=>8; pad = ((k÷2)*3,0), dilation = 3)
        c4 = Flux.DepthwiseConv((k,1), 2=>8; pad = ((k÷2)*4,0), dilation = 4)
        @λ(x -> cat(c1(x), c2(x), c3(x), c4(x); dims = 3))
    end

    model = Flux.Chain(
        CatConvLayers,
        Flux.BatchNorm(32, Flux.relu),
        Flux.Conv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # Flux.Conv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # Flux.Conv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.BatchNorm(32, Flux.relu),
        # Flux.Conv((3,1), 32=>32; pad = (1,0), stride = 2),
        # Flux.MaxPool((3,1)),
        # Flux.MeanPool((3,1)),
        # Flux.Dropout(0.3),
        DenseResize(),
        Flux.Dense(512, Nout),
        ParamsScale(),
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
        )),
    )

    return model
end

function forward_physics(x::Matrix{T}) where {T}
    rT1, nTE = T(1_000e-3 / 10e-3), 32 # Fixed params
    Smw  = zeros(Vec{3,T}, nTE) # Buffer for myelin signal
    Siew = zeros(Vec{3,T}, nTE) # Buffer for IE water signal
    M    = zeros(T, nTE, size(x,2)) # Total signal magnitude
    for j in 1:size(x,2)
        mwf, iewf, rT2iew, rT2mw, alpha = x[1,j], x[2,j], x[3,j], x[4,j], x[5,j]
        rT1iew, rT1mw = x[7,j], x[8,j]
        # rT1iew, rT1mw = rT1, rT1
        forward_prop!(Smw,  rT2mw,  rT1mw,  alpha, nTE)
        forward_prop!(Siew, rT2iew, rT1iew, alpha, nTE)
        @views M[:,j] .= norm.(transverse.(mwf .* Smw .+ iewf .* Siew))
    end
    return M
end
ForwardProp() = @λ(x -> forward_physics(x))

function basic_height32_generator_1(settings)
    T      :: Type   = settings["prec"] == 64 ? Float64 : Float32
    VT     :: Type   = settings["prec"] == 64 ? Vector{Float64} : Vector{Float32}
    H      :: Int    = settings["data"]["height"] :: Int # Data height
    C      :: Int    = settings["data"]["channels"] :: Int # Number of channels
    Nout   :: Int    = settings["model"]["Nout"] :: Int # Number of outputs
    type   :: Symbol = settings["model"]["resnet"]["type"] :: String |> Symbol # Factory type for BatchConvConnection
    Nfilt  :: Int    = settings["model"]["resnet"]["Nfilt"] :: Int # Number of features to upsample to from 1-feature input
    scale  :: VT     = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset :: VT     = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    
    θ = Nout
    @assert H == 32 # Model height should be 32

    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset) # Scales approx [-1,1] to param range
    InverseParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> inv.(scale), initβ = (args...) -> -offset./scale) # Scales param range to approx [-1,1]

    EvenSigns = cospi.(0:31) |> VT
    OddSigns = cospi.(1:32) |> VT
    function Corrections(F = 64, C = 8, k = 3, p = 5, σ = Flux.relu)
        Flux.Chain(
            InverseParamsScale(),
            Flux.Dense(θ, F, σ),
            ChannelResize(1),
            Flux.Conv((1,1), 1=>C; pad = (0,0)),
            # Flux.Dense(θ, 32*C, σ),
            # ChannelResize(C),
            IdentitySkip(
                Flux.Chain(
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    @λ(x -> σ.(x)),
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C),
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C, σ),
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    # Flux.BatchNorm(C),
                ),
            ),
            Flux.Conv((k,1), 1C=>2C; pad = (k÷2,0)),
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            IdentitySkip(
                Flux.Chain(
                    Flux.Conv((k,1), 2C=>2C; pad = (k÷2,0)),
                    @λ(x -> σ.(x)),
                    Flux.Conv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C),
                    # Flux.Conv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C, σ),
                    # Flux.Conv((k,1), 2C=>2C; pad = (k÷2,0)),
                    # Flux.BatchNorm(2C),
                ),
            ),
            Flux.Conv((k,1), 2C=>4C; pad = (k÷2,0)),
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            # Flux.Conv((1,1), C=>1; pad = (0,0)),
            DenseResize(),
            Flux.Dense((F÷16)*4C, 32),
            # @λ(x -> EvenSigns .* x), # Corrections typically alternate sign
        )
    end
    
    model = Flux.Chain(
        Sumout(
            ForwardProp(),
            Corrections(),
        ),
        @λ(x -> Flux.relu.(x)),
    )

    return model
end

function basic_height32_generator_2(settings)
    T      :: Type   = settings["prec"] == 64 ? Float64 : Float32
    VT     :: Type   = settings["prec"] == 64 ? Vector{Float64} : Vector{Float32}
    H      :: Int    = settings["data"]["height"] :: Int # Data height
    C      :: Int    = settings["data"]["channels"] :: Int # Number of channels
    Nout   :: Int    = settings["model"]["Nout"] :: Int # Number of outputs
    type   :: Symbol = settings["model"]["resnet"]["type"] :: String |> Symbol # Factory type for BatchConvConnection
    Nfilt  :: Int    = settings["model"]["resnet"]["Nfilt"] :: Int # Number of features to upsample to from 1-feature input
    scale  :: VT     = settings["model"]["scale"] :: Vector |> VT # Parameter scales
    offset :: VT     = settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    
    θ = Nout
    @assert H == 32 # Model height should be 32

    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset) # Scales approx [-1,1] to param range
    InverseParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> inv.(scale), initβ = (args...) -> -offset./scale) # Scales param range to approx [-1,1]

    OddSigns  = 1:H .|> n -> isodd(n)  ? one(T) : -one(T)
    EvenSigns = 1:H .|> n -> iseven(n) ? one(T) : -one(T)
    function ResRefinement(F = 32, C = 16, R = 4, k = 3, σ = Flux.relu)
        Flux.Chain(
            [IdentitySkip(
                Flux.Chain(
                    ChannelResize(1),
                    Flux.Conv((1,1), 1=>C; pad = (0,0)), # Upsample channels
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    # @λ(x -> σ.(x)),
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C, σ),
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    Flux.Conv((1,1), C=>1; pad = (0,0)), # Downsample channels
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
            Flux.Conv((1,1), 1=>C; pad = (0,0)),
            [IdentitySkip(
                Flux.Chain(
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    # @λ(x -> σ.(x)),
                    # Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C, σ),
                    Flux.Conv((k,1), C=>C; pad = (k÷2,0)),
                    Flux.BatchNorm(C),
                ),
            ) for i in 1:R]...,
            Flux.MeanPool((p,1); pad = (p÷2,0), stride = 4),
            DenseResize(),
            Flux.Dense((F÷4)*C, H),
            @λ(x -> OddSigns .* x), # Corrections typically alternate sign
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
            @λ(x -> OddSigns .* x), # Corrections typically alternate sign
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
        ForwardProp() |> RecurseCorrect,
        # ForwardProp() |> RecurseCorrect |> RecurseCorrect |> RecurseCorrect |> RecurseCorrect,
        @λ(x -> Flux.relu.(x)),
    )

    return model
end

function basic_DCGAN_1(::Type{T} = Float32;
        H      :: Int = 32, #settings["data"]["height"] :: Int # Data height
        C      :: Int = 1,  #settings["data"]["channels"] :: Int # Number of channels
        Nout   :: Int = 8,  #settings["model"]["Nout"] :: Int # Number of outputs
        # scale  :: VT  = 32, #settings["model"]["scale"] :: Vector |> VT # Parameter scales
        # offset :: VT  = 32, #settings["model"]["offset"] :: Vector |> VT # Parameter offsets
    ) where {T}
    
    @assert H == 32 # Model height should be 32
    @assert mod(H, 8) == 0 # Must be divisible by 8
    θ = Nout # Number of physical parameters

    ParamsScale() = Flux.Diagonal(scale, offset) # Scales approx [-0.5,0.5] to param range
    InverseParamsScale() = Flux.Diagonal(inv.(scale), .-offset./scale) # Scales param range to approx [-1,1]

    OddSigns  = 1:H .|> n -> isodd(n)  ? one(T) : -one(T)
    EvenSigns = 1:H .|> n -> iseven(n) ? one(T) : -one(T)

    function Generator()
        C,  k  = 32, (3,1)
        s1, p1 = (1,1), (1,1,0,0)
        s2, p2 = (2,1), (1,0,0,0) # Note: asymmetric padding
        return Sumout(
            ForwardProp(),
            Flux.Chain(
                InverseParamsScale(),
                Flux.Dense(θ, (H÷8) * C),
                ChannelResize(C),
                Flux.ConvTranspose(k, C => C÷2; pad = p2, stride = s2), # Output height: H÷4
                # Flux.BatchNorm(C÷2),
                @λ(x -> Flux.relu.(x)),
                Flux.ConvTranspose(k, C÷2 => C÷4; pad = p2, stride = s2), # Output height: H÷2
                # Flux.BatchNorm(C÷4),
                @λ(x -> Flux.relu.(x)),
                Flux.ConvTranspose(k, C÷4 => 1; pad = p2, stride = s2), # Output height: H
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
            Flux.Conv(k, 1 => C; pad = p, stride = s1), # Output height: H
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            Flux.Conv(k, C => 2C; pad = p, stride = s2), # Output height: H÷2
            # Flux.BatchNorm(2C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            Flux.Conv(k, 2C => 4C; pad = p, stride = s2), # Output height: H÷4
            # Flux.BatchNorm(4C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            Flux.Conv(k, 4C => 8C; pad = p, stride = s2), # Output height: H÷8
            # Flux.BatchNorm(8C),
            @λ(x -> Flux.leakyrelu.(x, T(0.2))),
            DenseResize(),
            Flux.Dense(8C * (H÷8), 1, Flux.sigmoid),
        )
    end

    sampler = ParamsScale()
    Z = B -> sampler(rand(T, θ, B) .- T(0.5))
    G = Flux.Chain(Generator(), @λ(x -> Flux.relu.(x)))
    D = Discriminator()

    return @ntuple(G, D, Z)
end
