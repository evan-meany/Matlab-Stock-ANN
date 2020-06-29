function network = initNet(inputs, expected, n_hidden, n_outputs)
    for i = 1 : n_hidden
        for j = 1 : length(inputs(1,:)) + 1
            hidden_layer.weights(i, j) = 2*rand() - 1; 
        end
    end
    for i = 1 : n_outputs
        for j = 1 : n_hidden + 1
            output_layer.weights(i, j) = 2*rand() - 1;
        end
    end
    network.inputs = inputs;
    network.expected = expected;
    network.hidden = hidden_layer;
    network.output = output_layer;
end