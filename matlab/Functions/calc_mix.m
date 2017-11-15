function P = calc_mix(Mset,n_Mset,T,MID_list,beta_value,selfU,neighborList,directionList)
U = selfU;
n_neighbor = length(neighborList);
M_center = ones(n_neighbor,1)*Mset;
M_neighbor = MID_list(neighborList)*ones(1,n_Mset);
is_zero = (M_center == M_neighbor) | isnan(M_neighbor);
M_beta = beta_value(directionList)*ones(1,n_Mset); % beta is n_direction-by-1 vector.
M_beta(is_zero) = 0;
U = U + sum(M_beta);
P = exp(-U/T)/sum(exp(-U/T));
end