% calculate sum rate for ARA (Heuristic method)
UE_num_list = 10:10:50;
AP_num = 26; % change this parameter
max_SINR = 66.24;
min_SINR = -10;
max_R = 500;
min_R = 10;
Nf = 3;
Thr_list = zeros(1, 5);
Fairness_list = zeros(1, 5);
for i = 1:length(UE_num_list)
    UE_num = UE_num_list(i);
    % change dataset path here
    input_name = ['dataset/25LiFi/' num2str(UE_num) 'UE_3Nf_withR/input_MixedGraph_5000.csv'];
    output_name = ['dataset/25LiFi/' num2str(UE_num) 'UE_3Nf_withR/output_MixedGraph_5000.csv'];
    dataset = csvread(input_name);
    %
    Thr_now_list = zeros(1, 5000);
    fairness_list = zeros(1, 5000);
    for j = 1:5000
        dataset_now = dataset(:, j);
        dataset_now = reshape(dataset_now, AP_num+1, UE_num);
        SINR = dataset_now(1:AP_num, :); % in dB
        R = dataset_now(AP_num+1, :); % in Mbps
        capacity = 20.*log2(1 + 10.^(SINR/10)); 
        X_iu = SSS(SINR, Nf);
        ARA_rho = ARA(X_iu, AP_num, UE_num);
        Thr_now_list(j) = sum(min(sum(capacity.*ARA_rho), R));
        fairness_list(j) = Jain_Fairness(capacity, ARA_rho, R, UE_num);
    end
    Thr_list(i) = sum(Thr_now_list)/5000;
    Fairness_list(i) = sum(fairness_list)/5000;
end
Thr_list;
Fairness_list;

function X_iu = SSS(SNR, N_f)
% input_data = SNR matrix
% outpot_data = X_iu matrix;
X_iu = zeros(size(SNR));
X_iu(1,:) = 1;
    SNR_LiFi = SNR(2:end, :);
    for i = 1:size(SNR_LiFi, 2)
        sorted_list = sort(SNR_LiFi(:, i), 'descend');
        for j = 1:N_f-1
            row = SNR(:, i) == sorted_list(j);
            X_iu(row, i) = 1;
        end
    end
end

function Rho = ARA(X_iu, AP_num, UE_num)
    Rho = zeros(AP_num, UE_num);
    for i = 1:1:AP_num
        Rho(i, (X_iu(i, :) == 1)) = 1/sum(X_iu(i, :)); 
    end
end

function fairness = Jain_Fairness(Capacity, Rho, R, UE_num)
    sat_list = min(sum(Capacity.*Rho)./R, 1);
    data1 = sum(sat_list)^2;
    data2 = UE_num*sum(sat_list.^2);
    fairness = data1/data2;
end