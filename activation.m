function activation = activation(weights, inputs)
    activation = 0;
    for i = 1 : length(weights)-1; activation = activation + ...
            weights(i) * inputs(i); end
    activation = activation + weights(end);
end