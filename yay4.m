clc; clear; close all;

%% read in data
data = readtable('snp500.csv'); 
rawdata = table2timetable(data);
data = table2array(data(:,2:end));
len = length(data(:,1));

%% Indicators
%Relative Strength Index
rsi = rsindex(rawdata, 14);
%Moving Averages: Linear/Exponential
malin = movavg(rawdata, 'linear', 5);
maexp = movavg(rawdata, 'exponential', 7);
%Stochastic Oscillator
stoch = stochosc(rawdata);
%Negative Volume Index 10th Features
nvi = negvolidx(rawdata);
%Positive Volume Index 11th Features
pvi = posvolidx(rawdata);
%Accumulation/Distribution OSciallator 12th Features
ado = adosc(rawdata);
%Stochastic Oscillator 24
so = stochosc(rawdata);
%Acceleration between times 25
acc = tsaccel(rawdata);
%Momentum between times 26
mom = tsmom(rawdata);

%% Create training data: Inputs
%First 14 days NAN
rsi = timetable2table(rsi);
rsi = table2array(rsi(:,2));
%Linear moving average of the close price
malin = timetable2table(malin);
malin = table2array(malin(:,5));   
%Exponential moving average of the close price
maexp = timetable2table(maexp);
maexp = table2array(maexp(:,5));  
%Fast Percent
stoch = timetable2table(stoch);
stoch = table2array(stoch(:,2));    

nvi = timetable2table(nvi);
nvi = table2array(nvi);

pvi = timetable2table(pvi);
pvi = table2array(pvi);

ado = timetable2table(ado);
ado = table2array(ado);

so = timetable2table(so);
so = table2array(so(:,2));

acc = timetable2table(acc);
acc = table2array(acc(:,2));

mom = timetable2table(mom);
mom = table2array(mom(:,2));

inputs = normalize([rsi,malin,maexp,stoch]);

%% Create training data: Expected
%Determine buy or not buy
expected = zeros(len,1);
for i=1:len;if(data(i,4)-data(i,1))>0;expected(i)=1;end;end

%% Create network
network = initNet(inputs(15:115,:),expected(15:115),10,2);
network = trainNetwork(network,500,.6,2);
network = predict(network, inputs(:,:), expected(:,:));