function network = forwardProp(network, row)
    hiddenout = zeros(1, length(network.hidden.weights(:,1)));
    output = zeros(1, length(network.output.weights(:,1)));
    for i = 1 : length(network.hidden.weights(:,1))
       hiddenout(i) = transfer(activation(network.hidden.weights(i,:), ...
           network.inputs(row,:)));
    end
    network.hidden.output = hiddenout;
    for i = 1 : length(network.output.weights(:,1))
       output(i) = transfer(activation(network.output.weights(i,:), ...
           hiddenout));
    end
    network.output.output = output;
end