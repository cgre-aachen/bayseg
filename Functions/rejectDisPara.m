function [mu,SIGMA,beta]=rejectDisPara(n,d,n_Mset,y,P,P_star,mu,mu_star,SIGMA,SIGMA_star,beta,beta_star,prior_para)
% prior_para is a cell containing all porior parameters:
% prior_para{1} = mean_mu, k x d matrix 
% prior_para{2} = sigma_mu, d x d x k matrix
% prior_para{3} = mean_beta, a column vector with length(beta)
% prior_para{4} = sigma_beta, a column vector with length(beta)
% prior_para{5} = nu_SIGMA,  nu_SIGMA = d + 1, 1 x 1 real number, the degree of freedom in a
%                           invWshart distribution
% prior_para{6} = b_SIGMA, b_SIGMA is a k x d matrix
% prior_para{7} = kesi_SIGMA, kesi_SIGMA is a k x d matrix

%============ calculate the log likelihood =========================== 
f_y_old=zeros(n,n_Mset);
f_y_star=zeros(n,n_Mset);
parfor k=1:n_Mset
    f_y_old(:,k)=mvnpdf(y,mu(k,:),SIGMA(:,:,k));
    f_y_star(:,k)=mvnpdf(y,mu_star(k,:),SIGMA_star(:,:,k));
end
p_y_old=sum(P.*f_y_old,2);
p_y_star=sum(P_star.*f_y_star,2);
logLikelihood_old=nansum(log(p_y_old));
logLikelihood_star=nansum(log(p_y_star));
%============= calculate the log priror ==============================
logPrior_old = 0;
logPrior_star = 0;
idx_mu = any(mu ~= mu_star,2);
if any(idx_mu)
    logPrior_old = logPrior_old + log(mvnpdf(mu(idx_mu,:),prior_para{1}(idx_mu,:),prior_para{2}(:,:,idx_mu)));
    logPrior_star = logPrior_star + log(mvnpdf(mu_star(idx_mu,:),prior_para{1}(idx_mu,:),prior_para{2}(:,:,idx_mu)));
end
idx_beta = (beta ~= beta_star);
if any(idx_beta)
    logPrior_old = logPrior_old + log(normpdf(beta(idx_beta),prior_para{3}(idx_beta),prior_para{4}(idx_beta)));
    logPrior_star = logPrior_star + log(normpdf(beta_star(idx_beta),prior_para{3}(idx_beta),prior_para{4}(idx_beta)));
end
idx_SIGMA = any(any(SIGMA ~= SIGMA_star));
if any(idx_SIGMA)
    Lambda = diag(SIGMA(:,:,idx_SIGMA)).^(1/2);
    Lambda_star = diag(SIGMA_star(:,:,idx_SIGMA)).^(1/2);
    R = diag(1./Lambda)*SIGMA(:,:,idx_SIGMA)*diag(1./Lambda);
    R_star = diag(1./Lambda_star)*SIGMA_star(:,:,idx_SIGMA)*diag(1./Lambda_star);
    logP_R = -0.5*(prior_para{5}+d+1)*log(det(R))-prior_para{5}/2*sum(log(diag(inv(R))));
    logP_R_star = -0.5*(prior_para{5}+d+1)*log(det(R_star))-prior_para{5}/2*sum(log(diag(inv(R_star))));
    logP_Lambda = sum(log(normpdf(log(Lambda'),prior_para{6}(idx_SIGMA,:),prior_para{7}(idx_SIGMA,:))));
    logP_Lambda_star = sum(log(normpdf(log(Lambda_star'),prior_para{6}(idx_SIGMA,:),prior_para{7}(idx_SIGMA,:))));
    logPrior_old = logPrior_old + logP_R + logP_Lambda;
    logPrior_star = logPrior_star + logP_R_star + logP_Lambda_star;
end
%============== posterior likehood =======================================
logLikelihood_old = logLikelihood_old + logPrior_old;
logLikelihood_star = logLikelihood_star + logPrior_star;
%============== accept and reject sample =================================
if logLikelihood_star>logLikelihood_old    
    mu=mu_star;
    SIGMA=SIGMA_star;
    beta=beta_star;
else
    accceptRate=exp(logLikelihood_star-logLikelihood_old);
    r=rand(1,1);
    if r<=accceptRate        
        mu=mu_star;
        SIGMA=SIGMA_star;
        beta=beta_star;
    end
end

end