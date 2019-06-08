function get_model(settings, model_settings)
    model =
        settings["name"] == "Keras_1D_Seq_Class" ? keras_1D_sequence_classification(settings, model_settings) :
        error("Unknown model: " * settings["name"])
    return model
end

function keras_1D_sequence_classification(settings, model_settings)
    H = settings["data"]["nT2"] # data height
    C = settings["data"]["channels"] # number of channels

    @unpack Nfeat, Npool, Nkern, Nout = model_settings
    Npad = Nkern ÷ 2 # pad size
    Ndense = Nfeat[end] * ((H ÷ Npool) ÷ Npool)
    
    model = Chain(
        # Two convolution layers followed by max pooling
        Conv((Nkern,), C => Nfeat[1], pad = (Npad,), relu), # (H, 1, 1) -> (H, Nfeat[1], 1)
        Conv((Nkern,), Nfeat[1] => Nfeat[1], pad = (Npad,), relu), # (H, Nfeat[1], 1) -> (H, Nfeat[1], 1)
        MaxPool((Npool,)), # (H, Nfeat[1], 1) -> (H/Npool, Nfeat[1], 1)
    
        # Two more convolution layers followed by mean pooling
        Conv((Nkern,), Nfeat[1] => Nfeat[2], pad = (Npad,), relu), # (H/Npool, Nfeat[1], 1) -> (H/Npool, Nfeat[2], 1)
        Conv((Nkern,), Nfeat[2] => Nfeat[2], pad = (Npad,), relu), # (H/Npool, Nfeat[2], 1) -> (H/Npool, Nfeat[2], 1)
        MeanPool((Npool,)), # (H/Npool, Nfeat[2], 1) -> (H/Npool^2, Nfeat[2], 1)
    
        # Dropout layer
        Dropout(0.5),
    
        # Dense layer
        x -> reshape(x, :, size(x, 3)),
        Dense(Ndense, Nout, relu),
        x -> reshape(x, Nout, 1, :))
    
    return model
end