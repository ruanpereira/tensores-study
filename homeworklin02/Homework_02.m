%% Homework 02
clc;
clear all;
close all;

%% Problem 1
x = (-10:1:10).';
load("Homework_02_data.mat")

y_linear    = observation_linear;
y_quadratic = observation_quadratic;
y_cubic     = observation_cubic;

projection_mtx_1 = [ones(length(x),1), x];
projection_mtx_2 = [ones(length(x),1), x, x.^2];
projection_mtx_3 = [ones(length(x),1), x, x.^2, x.^3];

solution_1 = pinv(projection_mtx_1)*y_linear;
solution_2 = pinv(projection_mtx_2)*y_quadratic;
solution_3 = pinv(projection_mtx_3)*y_cubic;

y_linear_estimation = solution_1(1,1) + solution_1(2,1)*x;
y_quadratic_estimation = solution_2(1,1) + solution_2(2,1)*x + solution_2(3,1)*x.^2;
y_cubic_estimation = solution_3(1,1) + solution_3(2,1)*x + solution_3(3,1)*x.^2 + solution_3(4,1)*x.^3;

% Linear
z = figure('DefaultAxesFontSize',16);
txt = ['Estimated vector'];
plot(x,y_linear_estimation,'-o','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold on;
txt = ['Noisy vector'];
plot(x,y_linear,'d','color', [0.8500 0.3250 0.0980], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold off;
ay = ylabel(['SE','$(\hat{y})$'],'interpreter','latex');
set(ay,'FontSize',20);
legend_copy = legend("location", "best",'Interpreter','latex');
set (legend_copy, "fontsize", 12);
grid on;

% Quadratic
z = figure('DefaultAxesFontSize',16);
txt = ['Estimated vector'];
plot(x,y_quadratic_estimation,'-o','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold on;
txt = ['Noisy vector'];
plot(x,y_quadratic,'d','color', [0.8500 0.3250 0.0980], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold off;
ay = ylabel(['SE','$(\hat{y})$'],'interpreter','latex');
set(ay,'FontSize',20);
legend_copy = legend("location", "best",'Interpreter','latex');
set (legend_copy, "fontsize", 12);
grid on;

% Cubic
z = figure('DefaultAxesFontSize',16);
txt = ['Estimated vector'];
plot(x,y_cubic_estimation,'-o','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold on;
txt = ['Noisy vector'];
plot(x,y_cubic,'d','color', [0.8500 0.3250 0.0980], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold off;
ay = ylabel(['SE','$(\hat{y})$'],'interpreter','latex');
set(ay,'FontSize',20);
legend_copy = legend("location", "best",'Interpreter','latex');
set (legend_copy, "fontsize", 12);
grid on;'pdf';

%% Problem 2
clear all;

SNR = [0:5:30];
iter_max = 100;
for snr = 1:length(SNR)
    for iter = 1: iter_max
        x = randn(5,1) + 1i*randn(5,1);
        A = randn(10,5) + 1i*randn(10,5);
        y = A*x;

        aux = SNR(snr);
        % y = 10log10(x)
        % x = 10^(y/10)
        snr_linear = 10^(aux/10);
        var_noise = (norm(y,2).^2)./(snr_linear);
        noise = sqrt(var_noise/2)*(randn(10,1) + 1i*randn(10,1));

        y_noisy = y + noise;

        x_estimation = pinv(A)*y_noisy;

        error(snr,iter) = (norm(y - A*x_estimation))^2./(norm(y).^2);
    end
end

z = figure('DefaultAxesFontSize',16);
txt = ['LS'];
plot(SNR,10*log10(mean(error,2)),'-o','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold off;
ay = ylabel(['NMSE','$(\hat{y})$',' in dB'],'interpreter','latex');
set(ay,'FontSize',20);
legend_copy = legend("location", "best",'Interpreter','latex');
set (legend_copy, "fontsize", 12);
grid on;

%% Problem 3
clear all;

M = 100;
N = [100:100:1000];
for i = 1:length(N)
    cost(i,1) = max(M,N(i))*min(M,N(i))^2;
end

z = figure('DefaultAxesFontSize',16);
txt = ['$M = 10$'];
semilogy(N,cost,'-o','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 12, "DisplayName", txt);
hold off;
ay = ylabel(['SE','$(\hat{y})$'],'interpreter','latex');
set(ay,'FontSize',20);
legend_copy = legend("location", "best",'Interpreter','latex');
set (legend_copy, "fontsize", 12);
grid on;
xlim([100 1000]);
