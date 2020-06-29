function network = backPropError(network, expected)
    for i = 1 : length(network.output.output)
        network.output.error(i) = (expected(i) - ...
                network.output.output(i))*transferDerivative(network.output.output(i));
    end
    for i = 1 : length(network.hidden.output)
        error = 0;
        for j = 1 : length(network.output.output)
            error = error + (network.output.weights(j,i)*network.output.error(j));
        end
        network.hidden.error(i) = error*transferDerivative(network.hidden.output(i));
    end
    
end