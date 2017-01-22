%=========================================================================%
% NN Toolbox                                                              %
% Example 3                                                               %
% FNN                                                                     %
% 2) Diagnosis (M = malignant, B = benign)  M=1, B=2                      %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
input = load('wdbc.data3.txt');
% normalize 0~1
for i = 3:32
    input(:, i) = (input(:, i) - min(input(:, i))) ./ (max(input(:, i)) - min(input(:, i)));
end
%input network *3/5
%because data result random so divide by 2
P(1:284, 1:30) = input(1:284, 3:32); %284
T(1:284) = input(1:284, 2);

%Pt = [0 1 -2 3 -4 5 -6 7 -8 9 -10];
%Tt = [0 1 2 3 4 5 6 7 8 9 10];
n = 285;
Pt(1:n, 1:30) = input(285:569, 3:32);
Tt(1:n) = input(285:569, 2);
%%
% Create network
% Horizontal to map Target matrix, neuron, 
% 2 learning algorithm
net = newff(P', T, [5 5], {'tansig', 'logsig', 'purelin'}, 'trainbfg'); %Quasi-Newton 
%net = newff(P', T, [5 5], {'tansig', 'logsig', 'purelin'}, 'trainrp'); %resillent backpropagation

% net

%=========================================================================%
% Set training parameter values: net.trainParam                           %
%=========================================================================%
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.003;
net.trainParam.goal = 0;
net.trainParam.show = 25;
%=========================================================================%
% Train network
net = train(net, P', T);

% Test network
yt = sim(net, Pt');
plot(yt, 'o');
xlabel('sample');
ylabel('yt');
title('yt raw')

% classify
count = 0;
for i = 1:n
    if yt(i) < 1.5
		yt(i) = 1;
    elseif 1.5 < yt(i)
		yt(i) = 2;
    end
	
	ERR(i) = Tt(i) - yt(i);
	
    if ERR(i) == 0
		count = count + 1;
    end
end

accuracy = count / n * 100

figure(2)

plot(1:n, Tt);
hold on;
plot(1:n, yt, 'ro');
xlabel('sample');
ylabel('sort');
title(['test accuracy = ', num2str(accuracy), '%'])
% Error
error_t = mse(yt - Tt)