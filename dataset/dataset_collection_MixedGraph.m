%% PT-LiFi
%% generate dataet for 16 LiFi 10m
% input: SINRs; output: Rho
clear;
clc;
% paramaters setting
load env_16LiFi_10m.mat env
env.AP_num = 17; % 16 LiFi
env.UE_num = 15;
env.N_f = 3; % subflows
batch_num = 1; % number of different simulations
batch_size = 1000; 
AP_position = [5, 5, 0.5;
              1.25, 1.25, 3; 3.75, 1.25, 3; 6.25, 1.25, 3; 8.75, 1.25, 3;
              1.25, 3.75, 3; 3.75, 3.75, 3; 6.25, 3.75, 3; 8.75, 3.75, 3;
              1.25, 6.25, 3; 3.75, 6.25, 3; 6.25, 6.25, 3; 8.75, 6.25, 3;
              1.25, 8.75, 3; 3.75, 8.75, 3; 6.25, 8.75, 3; 8.75, 8.75, 3];
%% Generate training dataset / testing dataset
input_names = arrayfun(@(i)['dataset/15UE_3Nf_input_MixedGraph2_' num2str(i) '.csv'], 1:batch_num, 'un',0); % need revise here
output_names = arrayfun(@(i)['dataset/15UE_3Nf_output_MixedGraph2_' num2str(i) '.csv'], 1:batch_num, 'un',0); % need revise here
for i = 1:length(input_names)
    input_data = zeros((env.AP_num+1)*env.UE_num, batch_size); % SNR
    output_data = zeros(env.AP_num*env.UE_num, batch_size); % Rho
    for j = 1:batch_size
        % env.R = ones(1, env.UE_num)*1000e6; % 1000Mbps = no data rate requirement
        env.R = max(min(gamrnd(1, 100, 1, env.UE_num), 500), 10)*1e6;
        % generate random samples with different positions
        UE_position_now = zeros(env.UE_num, 3);
        UE_position_now(:, 1:2) = rand(env.UE_num, 2)*env.X_length;
        % use SINR_calculation to get SINR
        SINR = zeros(env.AP_num, env.UE_num);
        % Calculate SINR
        for ii = 1:env.UE_num
            UE_position = UE_position_now(ii, :);
            for jj = 1:env.AP_num
                if jj == 1
                    SINR(jj,ii) = SINR_calculation(AP_position(jj,:), UE_position, "WiFi", env.AP_num-1, jj); % dB scale
                else
                    SINR(jj,ii) = SINR_calculation(AP_position(jj,:), UE_position, "LiFi", env.AP_num-1, jj-1); % dB scale
                end
            end
        end
        env.Capacity = env.B.*log2(1 + 10.^(SINR/10));
        env.SINR = max(SINR, -10); % choose -10 dB as breakpoint for minimum SINR
        %
        env.X_iu = SSS(env.SINR, env.N_f);
        JRA_Rho_iu = JRA(env);
        % set SNRs of unconencted UEs being 0
        input_now = [env.SINR; env.R/1e6];
        input_data(:, j) = reshape(input_now, (env.AP_num+1)*env.UE_num, 1); % save SINR data
        output_data(:, j) = reshape(JRA_Rho_iu, env.AP_num*env.UE_num ,1);
        fprintf('batch number = %d ', i);
        fprintf('batch sequence = %d \n', j);
    end
    input_thisname = input_names{i};
    output_thisname = output_names{i};
    %
    csvwrite(input_thisname, input_data);  % save input training data
    csvwrite(output_thisname, output_data); % save output training data
end