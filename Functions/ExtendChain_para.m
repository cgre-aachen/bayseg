function seg=ExtendChain_para(seg,Ext_Chain_length)
% mu is k by d
% SIGMA is d by d by k

Mset= 1:seg.num_of_clusters;
k = seg.num_of_clusters;
[n,d] = size(seg.field_value);
num_of_beta = size(seg.beta_bin, 1);
%===== pre-allocation ==========
MC_inferred(n,Ext_Chain_length) = NaN;
U(n,Ext_Chain_length-1) = NaN;
mu_bin(k,d,Ext_Chain_length) = 0;
SIGMA_bin(d,d,k,Ext_Chain_length) = 0;
beta_bin(num_of_beta,Ext_Chain_length) = 0;

%===== initialization ==========
MC_inferred(:,1) = seg.MC_inferred(:,end);
mu_bin(:,:,1) = seg.mu_bin(:,:,end);
SIGMA_bin(:,:,:,1) = seg.SIGMA_bin(:,:,:,end);
beta_bin(:,1) = seg.beta_bin(:,end);
display(mu_bin(:,:,1));
display(SIGMA_bin(:,:,:,1));
display(beta_bin(:,1));

mean_mu = seg.mu_bin(:,:,1);
sigma_mu = zeros(d,d,k);
b_SIGMA = zeros(k,d);
initial_Sigma = SIGMA_bin(:,:,:,1);
for i = 1:k
    sigma_mu(:,:,i) = diag(100*ones(d,1));
    b_SIGMA(i,:) = log(diag(initial_Sigma(:,:,i)).^(1/2))';
end
mean_beta = seg.beta_bin(:,1);
sigma_beta = 100*ones(length(mean_beta),1);
kesi_SIGMA = 100*ones(k,d);
prior_para = {mean_mu,sigma_mu,mean_beta,sigma_beta,d+1,b_SIGMA,kesi_SIGMA};


%===== MCMC simulation ===========
for i=2:Ext_Chain_length    
    [MC_inferred(:,i),U(:,i-1),mu_bin(:,:,i),SIGMA_bin(:,:,:,i),beta_bin(:,i)]=scanner_para(seg.Element, ...
    	MC_inferred(:,i-1),Mset,1,seg.para_scanorder,seg.num_of_color,seg.field_value,mu_bin(:,:,i-1), ...
    	SIGMA_bin(:,:,:,i-1),beta_bin(:,i-1),prior_para);
    display(i);
    display(mu_bin(:,:,i));
    display(SIGMA_bin(:,:,:,i));
    display(beta_bin(:,i));
end


MC_inferred = cat(2,seg.MC_inferred,MC_inferred);
U = cat(2,seg.energy_bin,U);
mu_bin = cat(3,seg.mu_bin,mu_bin);
SIGMA_bin = cat(4,seg.SIGMA_bin,SIGMA_bin);
beta_bin = cat(2,seg.beta_bin,beta_bin);

% =========== Post_process ==================
Chain_length = size(MC_inferred,2);
startPoint = round(Chain_length/5);
[Prob,InfEntropy,TotalInfEntr]=PostEntropy(MC_inferred,Mset,startPoint);
[~,latent_field_est] = max(Prob,[],2);
latent_field_est(isnan(sum(Prob,2))) = NaN;
[MU_hat,MU_std] = CalCenter(mu_bin,startPoint);
[R,stds,COV_hat,COV_std] = CalCorrCoeff(SIGMA_bin,startPoint);

% ========== output ==========================
seg.latent_field_est = latent_field_est;
seg.InfEntropy = InfEntropy;
seg.MU_hat = MU_hat;
seg.COV_hat = COV_hat;
seg.MC_inferred = MC_inferred;
seg.mu_bin = mu_bin;
seg.SIGMA_bin = SIGMA_bin;
seg.beta_bin = beta_bin;
seg.Prob = Prob;    
seg.TotalInfEntr = TotalInfEntr;
seg.MU_std = MU_std;
seg.COV_std = COV_std;
seg.CorrCoeffMatrix = R;
seg.stdMatrix = stds;
seg.energy_bin = U;
seg.totalEnergy = nansum(U);
end