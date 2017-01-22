%=========================================================================%
% NN Toolbox                                                              %
% Example 4                                                               %
% FNN                                                                     %
% 9) CYT=1,NUC=2,MIT=3,ME3=4,ME2=5,ME1=6,EXC=7,VAC=8,POX=9,ERL=10         %
%=========================================================================%
clc;
clear all;
close all;
% Collect data
input = load('yeast.data4.txt');
% normalize 0~1
for i = 1:8
    input(:, i) = (input(:, i) - min(input(:, i))) ./ (max(input(:, i)) - min(input(:, i)));
end
%input network *3/5
%because data result random so divide by 2
P(1:742, 1:8) = input(1:742, 1:8); %742
T(1:742) = input(1:742, 9);

%Pt = [0 1 -2 3 -4 5 -6 7 -8 9 -10];
%Tt = [0 1 2 3 4 5 6 7 8 9 10];
n = 742;
Pt(1:n, 1:8) = input(743:1484, 1:8);
Tt(1:n) = input(743:1484, 9);
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
net.trainParam.epochs = 5000;
net.trainParam.lr = 0.001;
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
	if yt(i) < 2.5
		yt(i) = 1;
    elseif yt(i) < 3
		yt(i) = 2;
    elseif yt(i) < 3.5
		yt(i) = 3;
    elseif yt(i) < 4
		yt(i) = 4;
	elseif yt(i) < 4.5
		yt(i) = 5;
	elseif yt(i) < 5
		yt(i) = 6;
	elseif yt(i) < 5.5
		yt(i) = 7;
	elseif yt(i) < 6
		yt(i) = 8;
	elseif yt(i) < 6.5
		yt(i) = 9;
	else
		yt(i) = 10;
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