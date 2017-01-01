function [MC,mu_bin,SIGMA_bin,beta_bin]=GenChain_para(Element,Mset,Chain_length,para_scanorder,num_of_color,y,beta_ini)
% mu is k by d
% SIGMA is d by d by k
% beta is a column vector
k=length(Mset); % the number of labels
C=4.6151205;
d=size(y,2); % the number of features
n=length(Element.Color); % the number of elements
%===== pre-allocation ==========
MC(n,Chain_length) = NaN;
mu_bin(k,d,Chain_length)=0;
SIGMA_bin(d,d,k,Chain_length)=0;
beta_bin(length(beta_ini),Chain_length)=0;

%=============================GMM clustering================================
options = statset('MaxIter',1000,'TolFun',5e-5);
GMModel=fitgmdist(y,k,'Regularize',0.001,'Replicates',100,'Start','plus','Options',options);
%===========================================================================

MC(:,1)=cluster(GMModel,y);
mu_bin(:,:,1)=GMModel.mu;
SIGMA_bin(:,:,:,1)=GMModel.Sigma;
beta_bin(:,1) = beta_ini;
display(mu_bin(:,:,1));
display(SIGMA_bin(:,:,:,1));
display(beta_bin(:,1));

mean_mu = GMModel.mu;
sigma_mu = zeros(d,d,k);
b_SIGMA = zeros(k,d);
for i = 1:k
    sigma_mu(:,:,i) = diag(100*ones(d,1));
    b_SIGMA(i,:) = log(diag(GMModel.Sigma(:,:,i)).^(1/2))';
end
mean_beta = beta_ini;
sigma_beta = 100*ones(length(beta_ini),1);
kesi_SIGMA = 100*ones(k,d);
prior_para = {mean_mu,sigma_mu,mean_beta,sigma_beta,d+1,b_SIGMA,kesi_SIGMA};


for i=2:Chain_length
    %=================================
    if i<=100
        T=C/log(1+i);       
    else
        T=C/log(101);
    end
    %=================================
    [MC(:,i),mu_bin(:,:,i),SIGMA_bin(:,:,:,i),beta_bin(:,i)]=scanner_para(Element,MC(:,i-1),Mset,T,para_scanorder,num_of_color,y,mu_bin(:,:,i-1),SIGMA_bin(:,:,:,i-1),beta_bin(:,i-1),prior_para);
    display(i);
    display(mu_bin(:,:,i));
    display(SIGMA_bin(:,:,:,i));
    display(beta_bin(:,i));
end

end
