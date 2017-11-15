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
    function logP = logPrior_mu(mu_singleCluster,mean_mu_singleCluster,sigma_mu_singleCluster)        
        logP = log(mvnpdf(mu_singleCluster,mean_mu_singleCluster,sigma_mu_singleCluster));
    end

    function logP = logPrior_beta(beta_singleDirec,mean_beta_singleDirec,sigma_beta_singleDirec)
        logP = log(normpdf(beta_singleDirec,mean_beta_singleDirec,sigma_beta_singleDirec));
    end

    function logP = logPrior_SIGMA(SIGMA_singleCluster,d,nu_SIGMA,b_SIGMA_sinbleCluster,kesi_SIGMA_singleCluster)
        Lambda = diag(SIGMA_singleCluster).^(1/2);
        R = diag(1./Lambda)*SIGMA_singleCluster*diag(1./Lambda);
        logP_R = -0.5*(nu_SIGMA+d+1)*log(det(R))-nu_SIGMA/2*sum(log(diag(inv(R))));
        logP_Lambda = sum(log(normpdf(log(Lambda'),b_SIGMA_sinbleCluster,kesi_SIGMA_singleCluster)));
        logP = logP_R + logP_Lambda;
    end

logPrior_old = 0;
logPrior_star = 0;

idx_mu = any(mu ~= mu_star,2);
if any(idx_mu)    
    mean_mu_singleCluster = prior_para{1}(idx_mu,:);
    sigma_mu_singleCluster = prior_para{2}(:,:,idx_mu);
    logPrior_old = logPrior_old + logPrior_mu(mu(idx_mu,:),mean_mu_singleCluster,sigma_mu_singleCluster);
    logPrior_star = logPrior_star + logPrior_mu(mu_star(idx_mu,:),mean_mu_singleCluster,sigma_mu_singleCluster);
end

idx_SIGMA = any(any(SIGMA ~= SIGMA_star));
if any(idx_SIGMA)
    nu_SIGMA = prior_para{5};
    b_SIGMA_sinbleCluster = prior_para{6}(idx_SIGMA,:);
    kesi_SIGMA_singleCluster = prior_para{7}(idx_SIGMA,:);
    logPrior_old = logPrior_old + logPrior_SIGMA(SIGMA(:,:,idx_SIGMA),d,nu_SIGMA,b_SIGMA_sinbleCluster,kesi_SIGMA_singleCluster);
    logPrior_star = logPrior_star + logPrior_SIGMA(SIGMA_star(:,:,idx_SIGMA),d,nu_SIGMA,b_SIGMA_sinbleCluster,kesi_SIGMA_singleCluster);
end

idx_beta = (beta ~= beta_star);
if any(idx_beta)    
    mean_beta_singleDirec = prior_para{3}(idx_beta);
    sigma_beta_singleDirec = prior_para{4}(idx_beta);
    log_prior_beta_old = sum(logPrior_beta(beta(idx_beta),mean_beta_singleDirec,sigma_beta_singleDirec));
    log_prior_beta_star = sum(logPrior_beta(beta_star(idx_beta),mean_beta_singleDirec,sigma_beta_singleDirec));
    logPrior_old = logPrior_old + log_prior_beta_old;
    logPrior_star = logPrior_star + log_prior_beta_star;
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