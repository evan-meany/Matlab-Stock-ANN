function network = trainNetwork(network, epoch, l_rate, n_outputs)
    for i = 1 : epoch
        sum_error = 0;
        for j = 1 : length(network.expected)
            network = forwardProp(network, j);
            expected = zeros(n_outputs,1);
            expected(network.expected(j) + 1) = 1;
            for k = 1 : n_outputs
                sum_error = sum_error + ...
                    ((expected(k) - network.output.output(k))^2);
            end
            network = backPropError(network, expected);
            network = updateWeights(network, j, l_rate);
        end
%     fprintf('epoch: %d, error: %f\n',i,sum_error);
    end
end