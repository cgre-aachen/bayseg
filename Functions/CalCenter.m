function [MU_hat,MU_std] = CalCenter(mu_bin,startPoint)
% this function is used for calculating the average center and standard
% deviation of the centers

n_var = size(mu_bin,2);
n_cluster = size(mu_bin,1);
n_sample = size(mu_bin,3);
MU_hat(n_cluster,n_var) = 0;
MU_std(n_cluster,n_var) = 0;

for i = 1:n_cluster
    for j = 1:n_var
        MU_hat(i,j) = mean(mu_bin(i,j,startPoint:n_sample));
        MU_std(i,j) = std(mu_bin(i,j,startPoint:n_sample));
    end
end
end