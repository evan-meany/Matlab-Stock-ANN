function network = updateWeights(network, row, l_rate)
    for i = 1 : length(network.hidden.error)
        for j = 1 : length(network.inputs(1,:))
            network.hidden.weights(i,j) = network.hidden.weights(i,j) + ...
                    (l_rate*network.hidden.error(i)*network.inputs(row,j));
        end
        network.hidden.weights(i,end)= network.hidden.weights(i,end) + ...
            (l_rate*network.hidden.error(i));
    end
    for i = 1 : length(network.output.error)
        for j = 1 : length(network.output.weights(1,:))-1
            network.output.weights(i,j) = network.output.weights(i,j) + ...
                    (l_rate*network.output.error(i)*network.hidden.output(j));            
        end
        network.output.weights(i,end) = network.output.weights(i,end) + ...
            (l_rate*network.output.error(i));        
    end
end