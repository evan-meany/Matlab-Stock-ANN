clc; clear;

data = readmatrix('wheat.csv');
% data = [[2.7810836,2.550537003,0],
% 	[1.465489372,2.362125076,0],
% 	[3.396561688,4.400293529,0],
% 	[1.38807019,1.850220317,0],
% 	[3.06407232,3.005305973,0],
% 	[7.627531214,2.759262235,1],
% 	[5.332441248,2.088626775,1],
% 	[6.922596716,1.77106367,1],
% 	[8.675418651,-0.242068655,1],
% 	[7.673756466,3.508563011,1]];


inputs = normalize(data(1:200,1:7));
expected = data(1:200,8);
network = initNet(inputs,expected,5,3);
network = trainNetwork(network, 100, .5, 3);

inputs_prediction = normalize(data(1:210,1:7));
expected_prediction = data(1:210,8);
network = predict(network, inputs_prediction, expected_prediction);
