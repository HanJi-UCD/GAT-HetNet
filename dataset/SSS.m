function X_iu = SSS(SNR, N_f)
% input_data = SNR matrix
% outpot_data = X_iu matrix;
X_iu = zeros(size(SNR));
X_iu(1, :) = 1; % keep WiFi connected
    SNR_LiFi = SNR(2:end, :);
    for i = 1:size(SNR_LiFi, 2)
        sorted_list = sort(SNR_LiFi(:, i), 'descend');
        for j = 1:N_f-1
            row = SNR(:, i) == sorted_list(j);
            X_iu(row, i) = 1;
        end
    end
end

