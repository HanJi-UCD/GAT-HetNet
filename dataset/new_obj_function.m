function cost = new_obj_function(X, AP_num, UE_num, AP_sel, capacity, R_required)
% this object function considers maximizing the sum of log throughput
for UE_ind = 1 : UE_num
    coeff_ind = (UE_ind - 1) * AP_num + 1 ;
    if UE_ind == 1
        if sum(AP_sel(:, UE_ind)) ~= 0
            cost = log( min(sum( X(coeff_ind:(coeff_ind+AP_num-1)) .* AP_sel(:, UE_ind) .* capacity(:, UE_ind) )/R_required(UE_ind), 1 ) );
        else
            cost = sum( X(coeff_ind:(coeff_ind+AP_num-1)) ) * 0 ; 
        end
    elseif sum(AP_sel(:, UE_ind)) ~= 0
        cost = cost + log(min(sum( X(coeff_ind:(coeff_ind+AP_num-1)) .* AP_sel(:, UE_ind) .* capacity(:, UE_ind) )/R_required(UE_ind), 1)) ;
    else
        cost = cost + sum( X(coeff_ind:(coeff_ind+AP_num-1)) ) * 0 ; 
    end
end
cost = - cost ;