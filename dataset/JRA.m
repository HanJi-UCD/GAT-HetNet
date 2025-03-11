function [Rho_iu] = JRA(env)
 % iterative JRA solution for each AP, not global solution JRA
AP_num = env.AP_num;
UE_num = env.UE_num;
Rho_iu = zeros(AP_num, UE_num);
mask = repmat(eye(AP_num), 1, UE_num);
if isempty( find(env.X_iu, 1) ) == 0
    A = repmat(reshape(env.X_iu, 1, AP_num*UE_num), AP_num, 1) ;
    A = A .* mask ;
    B = ones(AP_num, 1) ;
    LB = zeros(AP_num*UE_num, 1) ;
    UB = ones(AP_num*UE_num, 1) ;
    X0 = (LB+UB)/2;
    options = optimoptions('fmincon','display', 'off',"Algorithm","sqp", ...
         "SubproblemAlgorithm","cg",'MaxFunctionEvaluations',10000,'StepTolerance', 1e-10); % sqp or interior-point
    X = fmincon( @(X) new_obj_function(X, AP_num, UE_num, env.X_iu, env.Capacity, env.R), X0, A, B, [], [], LB, UB,  [], options);
    Rho_iu = reshape(X, AP_num, UE_num) ;
end
Rho_iu = Rho_iu.*env.X_iu;
end

