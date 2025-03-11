function SINR = SINR_calculation(AP_position, UE_position, mode, AP_size, AP_index)
% AP index, ranging from 0-16
N_0 = 10^(-21); % Noise power spectral density: Watt/Hz
B = 20000000; 
if mode == "LiFi"
    if AP_size == 4
        signal = Signal_power_calculation(AP_position, UE_position, mode, AP_size, AP_index);
        interference = 0;
    elseif AP_size == 9
        AP_position_9 = [1.25, 1.25, 3; 3.75, 1.25, 3; 6.25, 1.25, 3; 
                         1.25, 3.75, 3; 3.75, 3.75, 3; 6.25, 3.75, 3;
                         1.25, 6.25, 3; 3.75, 6.25, 3; 6.25, 6.25, 3;];
        signal_power_list = zeros(1, AP_size);
        for i = 1:AP_size
            signal_power_now = Signal_power_calculation(AP_position_9(i, :), UE_position, mode, AP_size, i);
            signal_power_list(i) = signal_power_now;
            matrix = [0,0,0,0,0,1,1,0,0; 0,0,0,0,0,0,0,1,0; 0,0,0,1,0,0,0,0,1; 0,0,1,0,0,0,0,0,1; 0,0,0,0,0,0,0,0,0;
                 1,0,0,0,0,0,1,0,0; 1,0,0,0,0,1,0,0,0; 0,1,0,0,0,0,0,0,0; 0,0,1,1,0,0,0,0,0];
        end
            interference = sum(signal_power_list .* matrix(AP_index, :));
            signal = signal_power_list(AP_index);
    elseif AP_size == 16
        AP_position_16 = [1.25, 1.25, 3; 3.75, 1.25, 3; 6.25, 1.25, 3; 8.75, 1.25, 3;
                       1.25, 3.75, 3; 3.75, 3.75, 3; 6.25, 3.75, 3; 8.75, 3.75, 3;                        
                       1.25, 6.25, 3; 3.75, 6.25, 3; 6.25, 6.25, 3; 8.75, 6.25, 3;
                       1.25, 8.75, 3; 3.75, 8.75, 3; 6.25, 8.75, 3; 8.75, 8.75, 3]; % room size of 10 m
        signal_power_list = zeros(1, AP_size);
        for i = 1:AP_size
            signal_power_now = Signal_power_calculation(AP_position_16(i, :), UE_position, mode, AP_size, i);
            signal_power_list(i) = signal_power_now;
        end
        matrix = [0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0; 
                  0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1; 
                  0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0; 
                  0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0;
                  0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0; 
                  0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0; 
                  1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0; 
                  0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1;
                  1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0; 
                  0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1; 
                  0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0; 
                  0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0;
                  0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0; 
                  0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0; 
                  1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0; 
                  0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0];
        interference = sum(signal_power_list .* matrix(AP_index, :));
        signal = signal_power_list(AP_index);
    else
        % 25 lifi AP case
        AP_position_25 = [1.25, 1.25, 3; 3.75, 1.25, 3; 6.25, 1.25, 3; 8.75, 1.25, 3; 11.25, 1.25, 3;
                       1.25, 3.75, 3; 3.75, 3.75, 3; 6.25, 3.75, 3; 8.75, 3.75, 3;  11.25, 3.75, 3;                        
                       1.25, 6.25, 3; 3.75, 6.25, 3; 6.25, 6.25, 3; 8.75, 6.25, 3;  11.25, 6.25, 3;
                       1.25, 8.75, 3; 3.75, 8.75, 3; 6.25, 8.75, 3; 8.75, 8.75, 3;  11.25, 8.75, 3;
                       1.25, 11.25, 3; 3.75, 11.25, 3; 6.25, 11.25, 3; 8.75, 11.25, 3;  11.25, 11.25, 3]; % room size of 12.5 m
        signal_power_list = zeros(1, AP_size);
        for i = 1:AP_size
            signal_power_now = Signal_power_calculation(AP_position_25(i, :), UE_position, mode, AP_size, i);
            signal_power_list(i) = signal_power_now;
        end
        matrix = [0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1; 
                  0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0; 
                  0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0; 
                  0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0; 
                  1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1; 
                  0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0; 
                  0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0; 
                  1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1; 
                  0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0; 
                  0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0; 
                  1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1; 
                  0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0; 
                  0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0; 
                  0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0; 
                  1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1; 
                  0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0; 
                  0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0; 
                  1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1; 
                  0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0; 
                  0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0; 
                  1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1; 
                  0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0; 
                  0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0; 
                  0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0; 
                  1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0;];
        interference = sum(signal_power_list .* matrix(AP_index, :));
        signal = signal_power_list(AP_index);
    end
    SINR = signal / (N_0 * B + interference);
else
    N_WiFi = -174; % dBm/Hz
    N_WiFi = 10^(N_WiFi/10) / 1000;  % convert to W/Hz
    B_WiFi = 20000000; % 20 MHz
    signal = Signal_power_calculation(AP_position, UE_position, mode, AP_size, AP_index);
    SINR = signal / (N_WiFi * B_WiFi);
end    
    SINR = max(SINR, 0.01); % set SINR > 0, being meaningful when converting to dB scale    
    SINR = 10 * log10(SINR); % in dB scale
end

%%
function P_signal = Signal_power_calculation(AP_position, UE_position, mode, AP_size, AP_index)
    if mode == "LiFi"
        if AP_size == 4
            X_length = 5;
            Y_length = 5;
            P_mod_list = [2.4, 2.4, 2.4, 2.4]; % = 3W*0.8
            R_pd_list = [0.53, 0.53, 0.53, 0.53];
        elseif AP_size == 9
            X_length = 7.5;
            Y_length = 7.5;
            P_mod_list = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4];
            R_pd_list = [0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53];
        elseif AP_size == 16
            X_length = 10; % room size of 10 m
            Y_length = 10; % room size of 10 m
            P_mod_list = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]; % P = 3W
            R_pd_list = [0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53];
        else
            X_length = 12.5; 
            Y_length = 12.5; 
            P_mod_list = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]; % P = 3W
            R_pd_list = [0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53];
        end
        Z_height = 3;
        P_mod = P_mod_list(AP_index); % Modulated power for RGBY LED
        R_pd = R_pd_list(AP_index); % PD responsivity for RGBY LED
        Phi = pi/3; % semiangle: radian
        FOV = 80 * (pi/180); % FOV: radian
        n = 1.5; % Reflective index of concentrator
        A = 0.0001; % Detector area: m**2
        m = 1; % Lambertian order
        
        % LOS
        d_LOS = pdist2(AP_position, UE_position);
        cos_phi = (Z_height - UE_position(3)) / d_LOS;
        
        if abs(acos(cos_phi)) <= Phi
            H_LOS = (m+1) * A * n^2 * Z_height^(m+1) / (2 * pi * (sin(FOV))^2 * d_LOS^(m+3)); % correct          
        else
            H_LOS = 0;
        end
        
        % NLOS
        H_NLOS = Capacity_NLOS(AP_position(1), AP_position(2), Z_height, UE_position(1), UE_position(2), X_length, Y_length, Z_height); % call sub-function
        P_signal = (R_pd * P_mod * (H_LOS + H_NLOS))^2;

    else
        % WiFi
        d_LOS = pdist2(AP_position, UE_position);
        radiation_angle = acos(0.5 / d_LOS); % radian unit
        f = 2.4 * 1000000000; % carrier frequency, 2.4 GHz
        P_WiFi_dBm = 20;
        P_WiFi = 10^(P_WiFi_dBm/10) / 1000; % 20 dBm, convert to watts: 0.1 W
        
        % 20 dB loss for concrete wall attenuation
        L_FS = 20 * log10(d_LOS) + 20 * log10(f) + 20 - 147.5; % free space loss, unit: dB
        d_BP = 3; % breakpoint distance
        
        if d_LOS <= d_BP
            K = 1; % Ricean K-factor 
            X = 3; % the shadow fading before breakpoint, unit: dB        
            LargeScaleFading = L_FS + X;                 
        else
            K = 0;
            X = 5; % the shadow fading after breakpoint, unit: dB                 
            LargeScaleFading = L_FS + 35 * log10(d_LOS / d_BP) + X;
        end
        
        H_WiFi = sqrt(K / (K+1)) * (cos(radiation_angle) + 1i * sin(radiation_angle)) + sqrt(1 / (K+1)) * (1 / sqrt(2) * rand(1) + 1i / sqrt(2) * rand(1)); % WiFi channel transfer function                      
        channel =  abs(H_WiFi)^2 * 10^(-LargeScaleFading / 10); % WiFi channel gain   
        P_signal = P_WiFi * channel; % range of (1000, 100000000)    
    end
end

%%
function H_NLOS = Capacity_NLOS(x_AP, y_AP, z_AP, x_UE, y_UE, X_length, Y_length, Z_height)
% input x-y-z coordinat of APs to return channel gain H of NLOS
Phi = pi/3; % semiangle: radian
FOV = 80/180*pi; % FOV: radian
m = -1/(log2(cos(Phi))); % Lambertian order
A = 0.0001; % Detector area: m^2
n = 1.5; % Reflective index of concentrator
UE = [x_UE, y_UE, 0]; %UE Location
AP = [x_AP, y_AP, z_AP];
rho = 0.8; % reflection coefficient of room walls
% X_length=5; % m, room size
% Y_length=5; % m, room size
% Z_height=3; % m, room size
step = 0.1;   % <--- change from 0.2 to 0.1
Nx = X_length/step; Ny = Y_length/step; Nz = Z_height/step; % number of grid in each surface
X = linspace(0, X_length, Nx+1);
Y = linspace(0, Y_length, Ny+1);
Z = linspace(0, Z_height, Nz+1);
dA = 0.01; % reflective area of wall
H_NLOS_W1 = zeros(length(Y)-1,length(Z)-1);
H_NLOS_W2 = zeros(length(X)-1,length(Z)-1);
H_NLOS_W3 = zeros(length(Y)-1,length(Z)-1);
H_NLOS_W4 = zeros(length(X)-1,length(Z)-1);
for i = 1:1:length(X)-1
    for j = 1:1:length(Z)-1
        %% H11_NLOS of Wall 1 (Left), 1st reflection channel gain between AP1 and UE
        Refl_point_W1 = [0, (Y(i)+Y(i+1))/2, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W1); 
        % d2=pdist2(UE,Refl_point_W1); % pdist2 function is time-costing
        d1 = sqrt((AP(1) - Refl_point_W1(1))^2 + (AP(2) - Refl_point_W1(2))^2 + (AP(3) - Refl_point_W1(3))^2); 
        d2 = sqrt((UE(1) - Refl_point_W1(1))^2 + (UE(2) - Refl_point_W1(2))^2 + (UE(3) - Refl_point_W1(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W1(3) - AP(3))/d1;
        cos_alpha = abs(AP(1) - Refl_point_W1(1))/d1;
        cos_beta = abs(UE(1) - Refl_point_W1(1))/d2;
        cos_psi = abs(UE(3) - Refl_point_W1(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W1(i,j)=(m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W1(i,j)=0;      
             end
          else
                H_NLOS_W1(i,j)=0;  
          end
        %% H11_NLOS of Wall 2 (Front)
        Refl_point_W2=[(X(i)+X(i+1))/2, 0, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W2); 
        % d2=pdist2(UE,Refl_point_W2); 
        d1 = sqrt((AP(1)-Refl_point_W2(1))^2 + (AP(2)-Refl_point_W2(2))^2 + (AP(3)-Refl_point_W2(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W2(1))^2 + (UE(2)-Refl_point_W2(2))^2 + (UE(3)-Refl_point_W2(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W2(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W2(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W2(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W2(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W2(i,j) = (m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W2(i,j) = 0;      
             end
          else
                H_NLOS_W2(i,j) = 0;  
          end
        %% H11_NLOS of Wall 3 (Right)
        Refl_point_W3 = [X_length, (Y(i)+Y(i+1))/2, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W3); 
        % d2=pdist2(UE,Refl_point_W3); 
        d1 = sqrt((AP(1)-Refl_point_W3(1))^2 + (AP(2)-Refl_point_W3(2))^2 + (AP(3)-Refl_point_W3(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W3(1))^2 + (UE(2)-Refl_point_W3(2))^2 + (UE(3)-Refl_point_W3(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W3(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W3(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W3(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W3(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi) <= Phi
             if abs(acosd(cos_psi)/180*pi) <= FOV
                H_NLOS_W3(i,j) = (m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W3(i,j)=0;  
             end
          else
                H_NLOS_W3(i,j)=0;  
          end
        %% H11_NLOS of Wall 4 (Back)
        Refl_point_W4=[(X(i)+X(i+1))/2, Y_length, (Z(j)+Z(j+1))/2];
        % d1=pdist2(AP,Refl_point_W4); 
        % d2=pdist2(UE,Refl_point_W4);
        d1 = sqrt((AP(1)-Refl_point_W4(1))^2 + (AP(2)-Refl_point_W4(2))^2 + (AP(3)-Refl_point_W4(3))^2); 
        d2 = sqrt((UE(1)-Refl_point_W4(1))^2 + (UE(2)-Refl_point_W4(2))^2 + (UE(3)-Refl_point_W4(3))^2); % distance calculation in 3-D space
        cos_phi = abs(Refl_point_W4(3)-AP(3))/d1;
        cos_alpha = abs(AP(1)-Refl_point_W4(1))/d1;
        cos_beta = abs(UE(1)-Refl_point_W4(1))/d2;
        cos_psi = abs(UE(3)-Refl_point_W4(3))/d2; % /sai/
          if abs(acosd(cos_phi)/180*pi)<= Phi
             if abs(acosd(cos_psi)/180*pi)<= FOV
                H_NLOS_W4(i,j)=(m+1)*A*rho*dA*cos_phi^m*cos_alpha*cos_beta*cos_psi*n^2/(2*pi^2*d1^2*d2^2*(sin(FOV))^2);
             else
                H_NLOS_W4(i,j)=0;    
             end
          else
                H_NLOS_W4(i,j)=0;  
          end
    end
end
H_NLOS = H_NLOS_W1 + H_NLOS_W2 + H_NLOS_W3 + H_NLOS_W4; % matrix data
H_NLOS = sum(sum(H_NLOS));
end







