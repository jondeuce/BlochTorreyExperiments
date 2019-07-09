get_model(settings::Dict, model_settings = settings["model"]) =
    model_settings["name"] == "ConvResNet" ? conv_resnet(settings, model_settings) :
    model_settings["name"] == "DenseConvResNet" ? dense_conv_resnet(settings, model_settings) :
    model_settings["name"] == "Keras1DSeqClass" ? keras_1D_sequence_classification(settings, model_settings) :
    model_settings["name"] == "TestModel1" ? test_model_1(settings, model_settings) :
    model_settings["name"] == "TestModel2" ? test_model_2(settings, model_settings) :
    model_settings["name"] == "TestModel3" ? test_model_3(settings, model_settings) :
    error("Unknown model: " * model_settings["name"])

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
function keras_1D_sequence_classification(settings, model_settings = settings["model"])
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

function test_model_1(settings, model_settings = settings["model"])
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

function test_model_2(settings, model_settings = settings["model"])
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

function test_model_3(settings, model_settings = settings["model"])
    H      = settings["data"]["height"] :: Int # Data height
    C      = settings["data"]["channels"] :: Int # Number of channels
    Nout   = model_settings["Nout"] :: Int # Number of outputs
    BN     = model_settings["batchnorm"] :: Bool # Use batch normalization
    scale  = model_settings["scale"] :: Vector # Parameter scales
    NP     = settings["data"]["preprocess"]["wavelet"]["apply"] == true ? length(scale) : 0

    PhysicalParams() = @λ(x -> x[1:NP, 1:C, :])
    NonPhysicsCoeffs() = @λ(x -> x[NP+1:end, 1:C, :])
    ParamsScale() = model_settings["scale"] == false ? identity : Scale(model_settings["scale"])
    MakeDropout() = model_settings["dropout"] == true ? Flux.AlphaDropout(0.5) : identity
    MakeActfun() = model_settings["act"] |> get_activation
    ResidualBlock() = IdentitySkip(DenseResConnection(4, 4, MakeActfun(); groupnorm = BN, mode = :post))

    # NonPhysicsCoeffs -> Output Parameters
    residualnetwork = Flux.Chain(
        NonPhysicsCoeffs(),
        DenseResize(),
        Flux.Dense(H*C - NP, 16, MakeActfun()),
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

function conv_resnet(settings, model_settings = settings["model"])
    H       = settings["data"]["height"] :: Int # Data height
    C       = settings["data"]["channels"] :: Int # Number of channels
    Nout    = model_settings["Nout"] :: Int # Number of outputs
    scale   = model_settings["scale"] :: Vector # Parameter scales
    DP      = model_settings["resnet"]["dropout"] :: Bool # Use batch normalization
    BN      = model_settings["resnet"]["batchnorm"] :: Bool # Use batch normalization
    GN      = model_settings["resnet"]["groupnorm"] :: Bool # Use group normalization
    Nkern   = model_settings["resnet"]["Nkern"] :: Int # Kernel size
    Nfeat   = model_settings["resnet"]["Nfeat"] :: Int # Kernel size
    Nconv   = model_settings["resnet"]["Nconv"] :: Int # Num residual connection layers
    Nblock  = model_settings["resnet"]["Nblock"] :: Int # Num residual connection layers
    @assert !(BN && GN)

    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> zeros(eltype(scale), size(scale)))
    MakeDropout() = DP ? Flux.AlphaDropout(0.5) : identity
    MakeActfun() = model_settings["act"] |> get_activation
    Upsample() = Flux.Conv((1,), 1 => Nfeat, MakeActfun(); pad = (0,)) # 1x1 upsample convolutions
    Downsample() = Flux.Conv((1,), Nfeat => 1, MakeActfun(); pad = (0,)) # 1x1 downsample convolutions
    ResidualBlock() = IdentitySkip(
        ConvResConnection(Nkern, Nfeat => Nfeat, MakeActfun();
        numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = :post))

    # Residual network
    residualnetwork = Flux.Chain(
        DenseResize(),
        Flux.Dense(H*C, H*C ÷ 2, MakeActfun()),
        ChannelResize(1),
        Upsample(),
        (ResidualBlock() for _ in 1:Nblock)...,
        MakeDropout(),
        Downsample(),
        DenseResize(),
        Flux.Dense(H*C ÷ 2, Nout),
    )

    # Output (assumes 1:2 is mwf/iewf, 3:4 are positive params, 5:6 is learned on a log scale)
    model = Flux.Chain(
        residualnetwork,
        # ParamsScale(), # Scale to parameter space
        @λ(x -> vcat(
            Flux.softmax(x[1:2,:]), # Automatically scales fractions
            Flux.relu.(x[3:4,:]) .* scale[3:4], # Params vary linearly over a range
            Flux.sigmoid.(x[5:6,:]) .* scale[5:6], # Params vary logarithmically over a range
        )),
    )

    return model
end

"""
Residual Dense Network for Image Super-Resolution: https://arxiv.org/abs/1802.08797
"""
function dense_conv_resnet(settings, model_settings = settings["model"])
    H       = settings["data"]["height"] :: Int # Data height
    C       = settings["data"]["channels"] :: Int # Number of channels
    Nout    = model_settings["Nout"] :: Int # Number of outputs
    scale   = model_settings["scale"] :: Vector # Parameter scales
    DP      = model_settings["resnet"]["dropout"] :: Bool # Use batch normalization
    BN      = model_settings["resnet"]["batchnorm"] :: Bool # Use batch normalization
    GN      = model_settings["resnet"]["groupnorm"] :: Bool # Use group normalization
    Nkern   = model_settings["resnet"]["Nkern"] :: Int # Kernel size
    Nfeat   = model_settings["resnet"]["Nfeat"] :: Int # Kernel size
    Nconv   = model_settings["resnet"]["Nconv"] :: Int # Num residual connection layers
    Nblock  = model_settings["resnet"]["Nblock"] :: Int # Num residual connection layers
    Ndense  = model_settings["resnet"]["Ndense"] :: Int # Num residual connection layers
    @assert !(BN && GN)

    ParamsScale() = Flux.Diagonal(Nout; initα = (args...) -> scale, initβ = (args...) -> zeros(eltype(scale), size(scale)))
    MakeDropout() = DP ? Flux.AlphaDropout(0.5) : identity
    MakeActfun() = model_settings["act"] |> get_activation
    Upsample() = Flux.Conv((1,), 1 => Nfeat, identity; pad = (0,)) # 1x1 upsample convolutions
    Downsample() = Flux.Conv((1,), Nfeat => 1, identity; pad = (0,)) # 1x1 downsample convolutions
    ResidualBlock() = IdentitySkip(
            ConvResConnection(
                Nkern, Nfeat => Nfeat, MakeActfun();
                numlayers = Nconv, batchnorm = BN, groupnorm = GN, mode = :post
            )
        )
    ResidualBlockChainFactory() = Flux.Chain(
            [ResidualBlock() for _ in 1:Nblock]...
        )
    DenseResidualNetwork() = DenseCatSkip(
            ResidualBlockChainFactory,
            (1,), Nfeat => Nfeat, identity;
            dims = 2, depth = Ndense
        )

    # Residual network
    denseresnet = Flux.Chain(
        DenseResize(),
        # Flux.Dense(H*C, H*C ÷ 2, MakeActfun()),
        ChannelResize(1),
        Upsample(),
        DenseResidualNetwork(),
        # MakeDropout(),
        Downsample(),
        DenseResize(),
        # Flux.Dense(H*C ÷ 2, Nout),
        Flux.Dense(H*C, Nout),
        # Flux.Diagonal(Nout),
    )

    # Output parameter handling:
    #   `relu` to force positivity
    #   `softmax` to force positivity and unit sum, i.e. fractions
    #   `sigmoid` to force positivity for parameters which vary over several orders of magnitude
    model = Flux.Chain(
        denseresnet,
        @λ(x -> vcat(
            # Flux.relu.(x[1:2,:]) .* scale[1:2], # Params vary linearly over a range
            # Flux.sigmoid.(x[3:3,:]) .* scale[3:3], # Params vary logarithmically over a range
            Flux.softmax(x[1:2,:]), # Automatically scales fractions
            Flux.relu.(x[3:4,:]) .* scale[3:4], # Params vary linearly over a range
            Flux.sigmoid.(x[5:5,:]) .* scale[5:5], # Params vary logarithmically over a range
        )),
    )

    return model
end