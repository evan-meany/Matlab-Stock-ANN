function network = predict(network, inputs, expected)
%     network.hidden.weights = [-1.482313569067226, 1.8308790073202204, 1.078381922048799; 0.23244990332399884, 0.3621998343835864, 0.40289821191094327];
%     network.output.weights = [2.5001872433501404, 0.7887233511355132, -1.1026649757805829; -2.429350576245497, 0.8357651039198697, 1.0699217181280656];
    wrong = 0;
    prediction = zeros(length(expected),1);
    tp = zeros(length(expected),1);
    tn = zeros(length(expected),1);
    fp = zeros(length(expected),1);
    fn = zeros(length(expected),1);
%     tpr = zeros(length(expected)/10,1);
%     fpr = zeros(length(expected)/10,1);
    for test = 1 : length(expected)
        hiddenout = zeros(1, length(network.hidden.weights(:,1)));
        output = zeros(1, length(network.output.weights(:,1)));
        for i = 1 : length(network.hidden.weights(:,1))
           hiddenout(i) = transfer(activation(network.hidden.weights(i,:), ...
               inputs(test,:)));
        end
        for i = 1 : length(network.output.weights(:,1))
           output(i) = transfer(activation(network.output.weights(i,:), ...
               hiddenout));
        end
        if max(output) == output(1)
            prediction(test) = 0;
            if expected(test) == 1
                wrong = wrong + 1;
                fn(test) = 1;
            else
                tn(test) = 1;
            end
        else
            prediction(test) = 1;
            if expected(test) == 1
                wrong = wrong + 1;
                tp(test) = 1;
            else
                fp(test) = 1;
            end
        end
%         fprintf('expected: %f; got: %f\n', expected(test), prediction(test)); 
        
    end
    fprintf('total wrong: %f; percent: %f \n', wrong, 100*wrong/length(expected));
    fprintf('accuracy: %f\n',100 - (100*wrong/length(expected)));
    %https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
    int_len = 50;
    interval = floor(length(expected)/int_len);
    exact = 1;
    if mod(length(expected), int_len) > 0
        interval = interval + 1;
        exact = 0;
    end
    tpr = zeros(interval,1);
    fpr = zeros(interval,1);
    
    switch exact
        case 0
            for i = 1 : interval-1
                tps = 0;
                tns = 0;
                fps = 0;
                fns = 0;
               for j = 1 : 50
                   tps = tp(((i-1)*int_len)+j) + tps;
                   tns = tn(((i-1)*int_len)+j) + tns;
                   fps = fp(((i-1)*int_len)+j) + fps;
                   fns = fn(((i-1)*int_len)+j) + fns;
               end
               tpr(i) = tps/(tps+fns);
               fpr(i) = fps/(fps+tns);
            end
            tps = 0;
            tns = 0;
            fps = 0;
            fns = 0;            
            for i = (interval-1)*int_len : length(expected)
               tps = tp(i) + tps;
               tns = tn(i) + tns;
               fps = fp(i) + fps;
               fns = fn(i) + fns;                
            end
            tpr(end) = tps/(tps+fns);
            fpr(end) = fps/(fps+tns);
        case 1
            
    end
    for i = 1 : length(tpr)
       if isnan(tpr(i))
           tpr(i) = 0;
       end
       if isnan(fpr(i))
          fpr(i) = 0; 
       end
    end
    
    figure
    sf = fit(fpr,tpr,'poly2');
    plot(sf,fpr,tpr)
    
end