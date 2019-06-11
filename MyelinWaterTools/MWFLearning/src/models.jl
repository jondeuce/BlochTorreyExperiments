function get_model(settings, model_settings = settings["model"])
    name = model_settings["name"]
    model =
        name == "Keras1DSeqClass" ? keras_1D_sequence_classification(settings, model_settings) :
        name == "TestModel1" ? test_model_1(settings, model_settings) :
        error("Unknown model: " * name)
    return model
end

get_activation(str::String) =
    str == "relu"      ? NNlib.relu :
    str == "sigma"     ? NNlib.σ :
    str == "leakyrelu" ? NNlib.leakyrelu :
    str == "elu"       ? NNlib.elu :
    str == "swish"     ? NNlib.swish :
    (@warn("Unkown activation function $str; defaulting to relu"); NNlib.relu)

function keras_1D_sequence_classification(settings, model_settings = settings["model"])
    H = settings["data"]["nT2"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Npool, Nkern, Nout, act = model_settings
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf2 * ((H ÷ Npool) ÷ Npool)
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
        x -> reshape(x, :, batchsize(x)),
        Flux.Dense(Ndense, Nout, actfun),
        model_settings["softmax"] ? NNlib.softmax : identity,
        )
    
    return model
end

function test_model_1(settings, model_settings = settings["model"])
    H = settings["data"]["nT2"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nf1, Nf2, Npool, Nkern, Nout, act = model_settings
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nf2 * ((H ÷ Npool) ÷ Npool)
    actfun = get_activation(act)
    
    model = Flux.Chain(
        # Two convolution layers followed by max pooling
        Flux.Conv((Nkern,), C => Nf1, pad = (Npad,), actfun), # (H, 1, 1) -> (H, Nf1, 1)
        Flux.Conv((Nkern,), Nf1 => Nf1, pad = (Npad,), actfun), # (H, Nf1, 1) -> (H, Nf1, 1)
        Flux.MaxPool((Npool,)), # (H, Nf1, 1) -> (H/Npool, Nf1, 1)
        Flux.BatchNorm(Nf1, actfun),
    
        # Two more convolution layers followed by mean pooling
        Flux.Conv((Nkern,), Nf1 => Nf2, pad = (Npad,), actfun), # (H/Npool, Nf1, 1) -> (H/Npool, Nf2, 1)
        Flux.Conv((Nkern,), Nf2 => Nf2, pad = (Npad,), actfun), # (H/Npool, Nf2, 1) -> (H/Npool, Nf2, 1)
        Flux.MaxPool((Npool,)), # (H/Npool, Nf2, 1) -> (H/Npool^2, Nf2, 1)
        Flux.BatchNorm(Nf2, actfun),
    
        # Dropout layer
        model_settings["dropout"] ? Flux.Dropout(0.5) : identity,
    
        # Dense + softmax layer
        x -> reshape(x, :, batchsize(x)),
        Flux.Dense(Ndense, Nout, actfun),
        Flux.BatchNorm(Nout, actfun),
        model_settings["softmax"] ? NNlib.softmax : identity,
        )
    
    return model
end