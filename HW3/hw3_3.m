clc;close all;clear all;

%% 產生數據
% train data

u = -2 + (2+2)*rand(1, 5000); %-2~2
for k = 5001:9000
    u(k) = 1.05 * sin(pi*k/45);
end

y(1) = 0, y(2) = 0, y(3) = 0;
for k = 3:8999
    y(k+1) = ( y(k)*y(k-1)*y(k-2)*u(k-1)*(y(k-2)-1)+u(k) ) / ...
			 ( 1 + y(k-2)^2 + y(k-1)^2 );
end

P_seq = con2seq(u);
T_seq = con2seq(y);

elmnet = newelm(P_seq, T_seq, 15, {'tansig','purelin'}, 'traingdm');
elmnet.trainParam.epochs = 100;
elmnet.trainParam.lr = 0.01;
elmnet.trainParam.goal = 0;
elmnet = train(elmnet, P_seq, T_seq, [], [], [], []);

% test data

for k = 1:1000
    if k <= 250
		u_test(k) = sin(pi*k/25);
	elseif k <= 500
		u_test(k) = 1.0;
	elseif k <= 750
		u_test(k) = -1.0;
	else
		u_test(k) = 0.3*sin(pi*k/25) + 0.1*sin(pi*k/32) + 0.6*sin(pi*k/10);
	end
end

y_test(1) = 0, y_test(2) = 0, y_test(3) = 0;
for k = 3:999
    y_test(k+1) = ( y_test(k)*y_test(k-1)*y_test(k-2)*u_test(k-1)*(y_test(k-2)-1)+u_test(k) ) / ...
			 ( 1 + y_test(k-2)^2 + y_test(k-1)^2 );
end

%compare to training neuro

Pt_seq = con2seq(u_test);
Yt_seq = sim(elmnet, Pt_seq);
Yt = cell2mat(Yt_seq)

figure(1)
plot(y_test(1,:), 'b');
axis([-inf inf -1.5 2]);
hold on;
plot(Yt(1,:), 'r');

error_t = mse(y_test-Yt);
title(['MSE=',num2str(error_t)]);