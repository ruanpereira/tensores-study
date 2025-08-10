load("ten_H")

R = ?;
SNR = 30;
[y,x] = tensor.ALS_estimation(ten_H,R,SNR);

z = figure('DefaultAxesFontSize',16);
semilogy(y,'-','color', [0 0.4470 0.7410], "linewidth", 3, "markersize", 6);
hold off;
title('Multilinear estimation performance')
ax = xlabel('ith iteration');
set(ax,'FontSize',20);
ay = ylabel('Error of reconstruction');
set(ay,'FontSize',20);
grid on;
