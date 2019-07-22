get_model(settings::Dict) =
    settings["model"]["name"] == "ConvResNet"       ?   conv_resnet(settings) :
    settings["model"]["name"] == "ResidualDenseNet" ?   residual_dense_net(settings) :
    settings["model"]["name"] == "Keras1DSeqClass"  ?   keras_1D_sequence_classification(settings) :
    settings["model"]["name"] == "TestModel1"       ?   test_model_1(settings) :
    settings["model"]["name"] == "TestModel2"       ?   test_model_2(settings) :
    settings["model"]["name"] == "TestModel3"       ?   test_model_3(settings) :
    settings["model"]["name"] == "TestModel4"       ?   test_model_4(settings) :
    settings["model"]["name"] == "ResNet"           ?   resnet(settings) :
    error("Unknown model: " * settings["model"]["name"])

get_activation(str::String) =
    str == "relu"      ? NNlib.relu :
    str == "sigma"     ? NNlib.σ :
    str == "leakyrelu" ? NNlib.leakyrelu :
    str == "elu"       ? NNlib.elu :
    str == "swish"     ? NNlib.swish :
    str == "softplus"  ? NNlib.softplus :
    (@warn("Unkown activation function $str; defaulting to softplus"); NNlib.softplus)

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
        ChannelwiseDense(H, C => C),
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
