get_model(settings::Dict, model_settings = settings["model"]) =
    model_settings["name"] == "Keras1DSeqClass" ? keras_1D_sequence_classification(settings, model_settings) :
    model_settings["name"] == "TestModel1" ? test_model_1(settings, model_settings) :
    model_settings["name"] == "TestModel2" ? test_model_2(settings, model_settings) :
    error("Unknown model: " * model_settings["name"])

get_activation(str::String) =
    str == "relu"      ? NNlib.relu :
    str == "sigma"     ? NNlib.σ :
    str == "leakyrelu" ? NNlib.leakyrelu :
    str == "elu"       ? NNlib.elu :
    str == "swish"     ? NNlib.swish :
    (@warn("Unkown activation function $str; defaulting to relu"); NNlib.relu)

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

        # Dense / batchnorm layer, but last actfun must be relu since outputs are positive
        model_settings["batchnorm"] ? Flux.Dense(Ndense ÷ 2, Nout, actfun) : Flux.Dense(Ndense ÷ 2, Nout, NNlib.relu),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nout, NNlib.relu) : identity,

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

    @unpack Nout, act, Nds = model_settings
    actfun = get_activation(act)

    model = Flux.Chain(
        DenseResize(),
        Flux.Dense(H * C, Nds[1], actfun),
        # model_settings["batchnorm"] ? Flux.BatchNorm(Nds[1], actfun) : identity,
        
        reduce(vcat, [
            Flux.Dense(Nds[i], Nds[i+1], actfun),
            # model_settings["batchnorm"] ? Flux.BatchNorm(Nds[i+1], actfun) : identity
        ] for i in 1:length(Nds)-1)...,

        Flux.Dense(Nds[end], Nout, actfun),
        model_settings["batchnorm"] ? Flux.BatchNorm(Nout, actfun) : identity,

        # Softmax
        model_settings["softmax"] ? NNlib.softmax : identity,
        
        # Swish to ensure positivity, unless softmax has already been applied
        model_settings["softmax"] ? identity : x -> NNlib.swish.(x),

        # Scale from (0,1) back to model parameter range
        model_settings["scale"] == false ? identity : Scale(model_settings["scale"]),
    )

    return model
end

function model_summary(model)
    @info "Model summary..."
    for layer in model
        if layer != identity
            print("        ")
            println(layer)
        end
    end
end