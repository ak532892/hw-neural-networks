%=========================================================================%
% NN Toolbox                                                              %
% Example 2                                                               %
% FNN                                                                     %
%                                                                         %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
input = load('wine.data2.txt');
% normalize 0~1
for i = 2:14
    input(:, i) = (input(:, i) - min(input(:, i))) ./ (max(input(:, i)) - min(input(:, i)));
end
%input network *3/5
P(1:35, 1:13) = input(1:35, 2:14); %35
P(36:77, 1:13) = input(60:101, 2:14); %42
P(78:105, 1:13) = input(131:158, 2:14); %28
T(1:35) = input(1:35, 1);
T(36:77) = input(60:101, 1);
T(78:105) = input(131:158, 1);

%Pt = [0 1 -2 3 -4 5 -6 7 -8 9 -10];
%Tt = [0 1 2 3 4 5 6 7 8 9 10];
n = 73;
Pt(1:24, 1:13) = input(36:59, 2:14);
Pt(25:53, 1:13) = input(102:130, 2:14);
Pt(54:n, 1:13) = input(159:178, 2:14);
Tt(1:24) = input(36:59, 1);
Tt(25:53) = input(102:130, 1);
Tt(54:n) = input(159:178, 1);
%%
% Create network
% Horizontal to map Target matrix, neuron, 
% 2 learning algorithm
net = newff(P', T, [10 10], {'tansig', 'logsig', 'purelin'}, 'trainbfg'); %Quasi-Newton 
%net = newff(P', T, [10 10], {'tansig', 'logsig', 'purelin'}, 'trainscg'); %Scaled Conjugate Gradient

% net

%=========================================================================%
% Set training parameter values: net.trainParam                           %
%=========================================================================%
net.trainParam.epochs = 3000;
net.trainParam.lr = 0.002;
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
    elseif (1.5 < yt(i) && yt(i) < 2.5)
		yt(i) = 2;
    elseif 2.5 < yt(i)
		yt(i) = 3;
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