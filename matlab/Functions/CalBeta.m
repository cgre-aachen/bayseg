function [beta_hat,beta_std]=CalBeta(beta_bin,startPoint)

n_sample = size(beta_bin,2);
beta_hat = mean(beta_bin(:,startPoint:n_sample),2);
beta_std = std(beta_bin(:,startPoint:n_sample),0,2);
end
