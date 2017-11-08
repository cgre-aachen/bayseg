function [mu,SIGMA,beta_value]=sampling_gmdisPara2(Element,MID_list,Mset,T,y,mu,SIGMA,beta_value,prior_para)
% mu is n_Mset by d
% SIGMA is d by d by k

SigmaProp_for_beta = diag(0.01*ones(length(beta_value),1));

num_of_ele = Element.num_of_elements;
n_Mset=size(mu,1);
d=size(mu,2);

%===== constract proposal function ===============    
beta_star = mvnrnd(beta_value',SigmaProp_for_beta)';

mu_star=zeros(n_Mset,d);
SIGMA_star=zeros(d,d,n_Mset);
sigma_jump(1,d,2) = 0;
sigma_jump(:,:,1) = 0.0005*ones(1,d);
sigma_jump(:,:,2) = 0.00005*ones(1,d);
Combinations = nchoosek(1:d,2);
n_axis = size(Combinations,1);

parfor i=1:n_Mset
    R = mvnrnd(zeros(1,d),sigma_jump);
    jump_mu = R(1,:);
    jump_logD = R(2,:);
    jump_theta = mvnrnd(zeros(1,n_axis),0.005*ones(1,n_axis));
    
    mu_star(i,:)=mu(i,:)+jump_mu;
    
    [V,D]=eig(SIGMA(:,:,i));    
    D_star = diag(exp(log(diag(D))+jump_logD'));
    
    A = eye(d);
    for j=1:n_axis
        Rotation_Matrix = rotation(V(:,Combinations(j,1)),V(:,Combinations(j,2)),jump_theta(j));
        A = Rotation_Matrix*A;
    end
    V_star = A*V;
    
    SIGMA_star(:,:,i)=V_star*D_star*V_star';
end

P=zeros(num_of_ele,n_Mset);
P_star = zeros(num_of_ele,n_Mset);
SU = Element.SelfU;
Nei = Element.Neighbors;
Direc = Element.Direction;
parfor idx=1:num_of_ele
    if ~isnan(MID_list(idx))
        P(idx,:) = calc_mix(Mset,n_Mset,T,MID_list,beta_value,SU(idx,:),Nei{idx},Direc{idx})
        P_star(idx,:) = calc_mix(Mset,n_Mset,T,MID_list,beta_star,SU(idx,:),Nei{idx},Direc{idx})
    end    
end
%===================================================================
for i=1:n_Mset
    mu_cadidate=mu;
    mu_cadidate(i,:)=mu_star(i,:);
    [mu,SIGMA,beta_value]=rejectDisPara(num_of_ele,d,n_Mset,y,P,P,mu,mu_cadidate,SIGMA,SIGMA,beta_value,beta_value,prior_para);
end

for i=1:n_Mset
    SIGMA_cadidate=SIGMA;
    SIGMA_cadidate(:,:,i)=SIGMA_star(:,:,i);
    [mu,SIGMA,beta_value]=rejectDisPara(num_of_ele,d,n_Mset,y,P,P,mu,mu,SIGMA,SIGMA_cadidate,beta_value,beta_value,prior_para);
end

[mu,SIGMA,beta_value]=rejectDisPara(num_of_ele,d,n_Mset,y,P,P_star,mu,mu,SIGMA,SIGMA,beta_value,beta_star,prior_para);

end