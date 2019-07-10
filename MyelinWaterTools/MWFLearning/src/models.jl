get_model(settings::Dict) =
    settings["model"]["name"] == "ConvResNet"       ?   conv_resnet(settings) :
    settings["model"]["name"] == "ResidualDenseNet" ?   residual_dense_net(settings) :
    settings["model"]["name"] == "Keras1DSeqClass"  ?   keras_1D_sequence_classification(settings) :
    settings["model"]["name"] == "TestModel1"       ?   test_model_1(settings) :
    settings["model"]["name"] == "TestModel2"       ?   test_model_2(settings) :
    settings["model"]["name"] == "TestModel3"       ?   test_model_3(settings) :
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
    model_settings = settings["model"]
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Npool, Nkern, Nout, act = model_settings
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf2 * (H ÷ Npool ÷ Npool)
    actfun = get_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling
        Flux.Conv((Nkern,), C => Nf1, pad = (Npad,), actfun), # (H, 1, 1) -> (H, Nf1, 1)
        Flux.Conv((Nkern,), Nf1 => Nf1, pad = (Npad,), actfun), # (H, Nf1, 1) -> (H, Nf1, 1)
        Flux.MaxPool((Npool,)), # (H, Nf1, 1) -> (H/Npool, Nf1, 1)

        # Two more convolution layers followed by mean pooling
        Flux.Conv((Nkern,), Nf1 => Nf2, pad = (Npad,), actfun), # (H/Npool, Nf1, 1) -> (H/Npool, Nf2, 1)
        Flux.Conv((Nkern,), Nf2 => Nf2, pad = (Npad,), actfun), # (H/Npool, Nf2, 1) -> (H/Npool, Nf2, 1)
        Flux.MeanPool((Npool,)), # (H/Npool, Nf2, 1) -> (H/Npool^2, Nf2, 1)

        # Dropout layer
        model_settings["dropout"] ? Flux.Dropout(0.5) : identity,

        # Dense + softmax layer
        DenseResize(),
        Flux.Dense(Ndense, Nout, actfun),
        model_settings["softmax"] ? NNlib.softmax : identity,
    )
    
    return model
end

function test_model_1(settings)
    model_settings = settings["model"]
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Nf3, Npool, Nkern, Nout, act = model_settings
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf3 * (H ÷ Npool ÷ Npool ÷ Npool)
    actfun = get_activation(act)

    model = Flux.Chain(
        # Two convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,), C => Nf1, pad = (Npad,), actfun),
        Flux.Conv((Nkern,), Nf1 => Nf1, pad = (Npad,), actfun),
        Flux.MaxPool((Npool,)),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nf1, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,), Nf1 => Nf2, pad = (Npad,), actfun),
        Flux.Conv((Nkern,), Nf2 => Nf2, pad = (Npad,), actfun),
        Flux.MaxPool((Npool,)),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nf2, actfun) : identity,

        # Two more convolution layers followed by max pooling and batch normalization
        Flux.Conv((Nkern,), Nf2 => Nf3, pad = (Npad,), actfun),
        Flux.Conv((Nkern,), Nf3 => Nf3, pad = (Npad,), actfun),
        Flux.MaxPool((Npool,)),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nf3, actfun) : identity,

        # Dropout layer
        model_settings["dropout"] ? Flux.Dropout(0.5) : identity,

        # Dense / batchnorm layer
        DenseResize(),
        Flux.Dense(Ndense, Ndense ÷ 2, actfun),
        model_settings["batchnorm"] ? Flux.BatchNorm(Ndense ÷ 2, actfun) : identity,

        # Dense / batchnorm layer, but last actfun must be softplus since outputs are positive
        model_settings["batchnorm"] ? Flux.Dense(Ndense ÷ 2, Nout, actfun) : Flux.Dense(Ndense ÷ 2, Nout, NNlib.softplus),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nout, NNlib.softplus) : identity,

        # Softmax
        model_settings["softmax"] ? NNlib.softmax : identity,

        # Scale from (0,1) back to model parameter range
        model_settings["scale"] == false ? identity : Scale(model_settings["scale"]),
    )

    return model
end

function test_model_2(settings)
    model_settings = settings["model"]
    H = settings["data"]["height"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nout, act, Nd = model_settings
    actfun = get_activation(act)

    model = Flux.Chain(
        DenseResize(),
        Flux.Dense(H * C, Nd[1], actfun),
        # model_settings["batchnorm"] ? Flux.BatchNorm(Nd[1], actfun) : identity,
        
        reduce(vcat, [
            Flux.Dense(Nd[i], Nd[i+1], actfun),
            # model_settings["batchnorm"] ? Flux.BatchNorm(Nd[i+1], actfun) : identity
        ] for i in 1:length(Nd)-1)...,

        Flux.Dense(Nd[end], Nout, actfun),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nout, actfun) : identity,

        # Softmax
        model_settings["softmax"] ? NNlib.softmax : identity,
        
        # Softplus to ensure positivity, unless softmax has already been applied
        model_settings["softmax"] ? identity : @λ(x -> NNlib.softplus.(x)),

        # Scale from (0,1) back to model parameter range
        model_settings["scale"] == false ? identity : Scale(model_settings["scale"]),
    )

    return model
end

function test_model_3(settings)
    model_settings = settings["model"]
    H      = settings["data"]["height"] :: Int # Data height
    C      = settings["data"]["channels"] :: Int # Number of channels
    actfun = model_settings["act"] |> get_activation
    Nout   = model_settings["Nout"] :: Int # Number of outputs
    BN     = model_settings["batchnorm"] :: Bool # Use batch normalization
    scale  = model_settings["scale"] :: Vector # Parameter scales
    NP     = settings["data"]["preprocess"]["wavelet"]["apply"] == true ? length(scale) : 0

    MakeActfun() = @λ x -> actfun.(x)
    PhysicalParams() = @λ(x -> x[1:NP, 1:C, :])
    NonPhysicsCoeffs() = @λ(x -> x[NP+1:end, 1:C, :])
    ParamsScale() = model_settings["scale"] == false ? identity : Scale(model_settings["scale"])
    MakeDropout() = model_settings["dropout"] == true ? Flux.AlphaDropout(0.5) : identity
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
    model_settings = settings["model"]
    H       = settings["data"]["height"] :: Int # Data height
    C       = settings["data"]["channels"] :: Int # Number of channels
    Nout    = model_settings["Nout"] :: Int # Number of outputs
    actfun  = model_settings["act"] |> get_activation # Activation function
    scale   = model_settings["scale"] :: Vector # Parameter scales
    offset  = model_settings["offset"] :: Vector # Parameter offsets
    DP      = model_settings["resnet"]["dropout"] :: Bool # Use batch normalization
    BN      = model_settings["resnet"]["batchnorm"] :: Bool # Use batch normalization
    GN      = model_settings["resnet"]["groupnorm"] :: Bool # Use group normalization
    Nkern   = model_settings["resnet"]["Nkern"] :: Int # Kernel size
    Nfeat   = model_settings["resnet"]["Nfeat"] :: Int # Kernel size
    Nconv   = model_settings["resnet"]["Nconv"] :: Int # Num residual connection layers
    Nblock  = model_settings["resnet"]["Nblock"] :: Int # Num residual connection layers
    @assert !(BN && GN)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)
    MakeDropout() = DP ? Flux.AlphaDropout(0.5) : identity
    Upsample() = Flux.Conv((1,), 1 => Nfeat, actfun; pad = (0,)) # 1x1 upsample convolutions
    Downsample() = Flux.Conv((1,), Nfeat => 1, actfun; pad = (0,)) # 1x1 downsample convolutions
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
        Flux.Dense(H*C ÷ 2, Nout),
    )

    # Output params
    model = Flux.Chain(
        residualnetwork,
        ParamsScale(), # Scale to parameter space
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
            # x[6:6,:], # Unbounded parameters
        )),
    )

    return model
end

"""
Residual Dense Network for Image Super-Resolution: https://arxiv.org/abs/1802.08797
"""
function residual_dense_net(settings)
    model_settings = settings["model"]
    H       = settings["data"]["height"] :: Int # Data height
    C       = settings["data"]["channels"] :: Int # Number of channels
    Nout    = model_settings["Nout"] :: Int # Number of outputs
    actfun  = model_settings["act"] |> get_activation # Activation function
    scale   = model_settings["scale"] :: Vector # Parameter scales
    offset  = model_settings["offset"] :: Vector # Parameter offsets
    DP      = model_settings["resnet"]["dropout"] :: Bool # Use batch normalization
    BN      = model_settings["resnet"]["batchnorm"] :: Bool # Use batch normalization
    GN      = model_settings["resnet"]["groupnorm"] :: Bool # Use group normalization
    mode    = model_settings["resnet"]["batchmode"] :: String |> Symbol # Batchnorm mode for BatchConvConnection
    Nkern   = model_settings["resnet"]["Nkern"] :: Int # Convolution kernel size
    Nconv   = model_settings["resnet"]["Nconv"] :: Int # Convolutions per BatchConvConnection
    Nfeat   = model_settings["resnet"]["Nfeat"] :: Int # Number of features to upsample to from 1-feature input
    Nblock  = model_settings["resnet"]["Nblock"] :: Int # Number of blocks in densely connected RDB layer
    Ndense  = model_settings["resnet"]["Ndense"] :: Int # Number of blocks in GlobalFeatureFusion concatenation layer
    @assert !(BN && GN)

    MakeActfun() = @λ x -> actfun.(x)
    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> offset)
    MakeDropout() = DP ? Flux.AlphaDropout(0.5) : identity
    Upsample(ch) = Flux.Conv((1,), ch, identity; pad = (0,)) # 1x1 upsample convolutions
    Downsample(ch) = Flux.Conv((1,), ch, identity; pad = (0,)) # 1x1 downsample convolutions
    
    function DFF()
        local G0, G, C, D, k = Nfeat, Nfeat, Nblock, Ndense, (Nkern,)
        ConvFactory = @λ ch -> Flux.Conv(k, ch, actfun; pad = (k.-1).÷2)
        BatchConvFactory = @λ ch -> BatchConvConnection(k, ch, actfun; numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = mode)
        # Factory = ConvFactory
        Factory = BatchConvFactory
        DenseFeatureFusion(Factory, G0, G, C, D, k, actfun; dims = 2)
    end

    # Residual network
    residualdensenet = Flux.Chain(
        # ChannelResize(4),
        # ChannelwiseDense(H*C ÷ 4, 4 => 1, actfun),
        ChannelResize(1),
        Upsample(1 => Nfeat),
        DFF(),
        # MakeDropout(),
        Flux.BatchNorm(Nfeat, actfun),
        # Flux.GroupNorm(Nfeat, Nfeat÷2, actfun),
        Downsample(Nfeat => 1),
        DenseResize(),
        # Flux.Dense(H*C ÷ 4, Nout),
        Flux.Dense(H*C, Nout),
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
            Flux.softmax(x[1:2,:]), # Positive fractions with unit sum
            Flux.relu.(x[3:5,:]), # Positive parameters
            x[6:6,:], # Unbounded parameters
        )),
    )

    return model
end