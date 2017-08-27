function beta = updateBeta(Element,Mset,T,MID_list,beta,SigmaProp)
% beta is a column vector.

%===== generate a candidate using Gaussian proposal =========
beta_star = mvnrnd(beta',SigmaProp)';
%===== calculate prior ======================================
if length(beta) == 4
    beta_mean = zeros(4,1); % beta is a column vector
    beta_sigma = diag(10*ones(4,1));
else
    beta_mean = zeros(13,1); % beta is a column vector
    beta_sigma = diag(13*ones(4,1));
end

logprior = log(mvnpdf(beta',beta_mean',beta_sigma));
logprior_star = log(mvnpdf(beta_star',beta_mean',beta_sigma));

%===== calsulate posteriori loglikelihood ===================
n_of_element = length(MID_list);
n_Mset = length(Mset);

loglike(n_of_element,1) = 0;
loglike_star(n_of_element,1) = 0;

temp_U = Element.SelfU;
temp_Nei = Element.Neighbors;
temp_Direc = Element.Direction;
parfor idx=1:n_of_element
    if ~isnan(MID_list(idx))
        U = temp_U(idx,:); % assign the energy of sigle site clique
        U_star = temp_U(idx,:);
        n_neighbor = length(temp_Nei{idx});
        M_center = ones(n_neighbor,1)*Mset;
        M_neighbor = MID_list(temp_Nei{idx})*ones(1,n_Mset);
        is_zero = (M_center==M_neighbor);
        M_beta = beta(temp_Direc{idx})*ones(1,n_Mset); % beta is n_direction-by-1 vector.
        M_beta_star = beta_star(temp_Direc{idx})*ones(1,n_Mset); % beta_star is n_direction-by-1 vector.
        M_beta(is_zero) = 0;
        M_beta_star(is_zero) = 0;
        U = U + sum(M_beta);
        U_star = U_star + sum(M_beta_star);
        P = exp(-U/T)/sum(exp(-U/T));
        P_star = exp(-U_star/T)/sum(exp(-U_star/T));
        loglike(idx) = log(P(MID_list(idx)));
        loglike_star(idx) = log(P_star(MID_list(idx)));
    end
end

logSum = sum(loglike) + logprior;
logSum_star = sum(loglike_star) + logprior_star;
%====== accept/reject beta ============================
if logSum_star > logSum
    beta=beta_star;
else
    acceptRate = exp(logSum_star-logSum);
    r=rand(1);
    if r <= acceptRate
        beta=beta_star;
    end
end
end