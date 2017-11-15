function BIC = calculateBIC(seg)

n = length(seg.latent_field_est);
n_Mset = seg.num_of_clusters;

P=zeros(n,n_Mset);
SU = seg.Element.SelfU;
Nei = seg.Element.Neighbors;
Direc = seg.Element.Direction;
MID_list = seg.latent_field_est;
beta_value = seg.beta_hat;
Mset = 1:n_Mset;
parfor idx=1:n
    if ~isnan(MID_list(idx))
        U=SU(idx,:); % assign the energy of sigle site clique
        U_star = SU(idx,:);
        n_neighbor=length(Nei{idx});
        M_center = ones(n_neighbor,1)*Mset;
        M_neighbor = MID_list(Nei{idx})*ones(1,n_Mset);
        is_zero = (M_center==M_neighbor) | isnan(M_neighbor);
        M_beta = beta_value(Direc{idx})*ones(1,n_Mset); % beta is n_direction-by-1 vector.        
        M_beta(is_zero) = 0;        
        U = U + sum(M_beta);        
        P(idx,:)=exp(-U)/sum(exp(-U));        
    end    
end

f_y = zeros(n,n_Mset);
mu = seg.MU_hat;
SIGMA = seg.COV_hat;
y = seg.field_value;
parfor k=1:n_Mset
    f_y(:,k)=mvnpdf(y,mu(k,:),SIGMA(:,:,k));    
end
p_y = sum(P.*f_y,2);
logLikelihood=nansum(log(p_y));
n_features = size(mu,2);
k = length(beta_value) + n_Mset*(n_features + n) + n_Mset*n_features*(n_features+1)/2;
BIC = -2*logLikelihood + k*log(n);
end