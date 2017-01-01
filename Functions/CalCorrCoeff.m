function [R,stds,COV_hat,COV_std] = CalCorrCoeff(SIGMA_bin,startPoint)
% this function is used for postprocess of the data from HMRF simulation
% SIGMA_bin is the 4-D matrix of the samples of the covariance matrix
% startPoint is the Nbr of the iteration for calculate the correlation
% coefficients

n_var = size(SIGMA_bin,1);
n_cluster = size(SIGMA_bin,3);
n_samples = size(SIGMA_bin,4);
R = zeros(n_var,n_var,n_cluster);
stds = zeros(n_var,n_cluster);
temp = SIGMA_bin(:,:,:,startPoint:n_samples);
COV_hat = zeros(n_var,n_var,n_cluster);
COV_std = zeros(n_var,n_var,n_cluster);
for i = 1:n_cluster    
    for m = 1:n_var
        for n = m:n_var         
            COV_hat(m,n,i) = mean(temp(m,n,i,:));            
            COV_hat(n,m,i) = COV_hat(m,n,i);
            COV_std(m,n,i) = std(temp(m,n,i,:));
            COV_std(n,m,i) = COV_std(m,n,i);
        end
    end
    [R(:,:,i),stds(:,i)] = corrcov(COV_hat(:,:,i));
end

end