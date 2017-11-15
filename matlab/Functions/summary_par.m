function [HMRF_matrix,GMM_matrix,MCR_HMRF,new_order] = summary_par(mu_bin,SIGMA_bin,beta_bin,MID_list_true,MC_inferred)
d = size(mu_bin,2);
k = size(mu_bin,1);
n_dispara = d*(d+3)/2;
HMRF_matrix = zeros(k*n_dispara+2,2);
GMM_matrix = zeros(k*n_dispara+1,1);
MU = [6,9;4,12;7,14];
%MU = [1 5.1 1;1.5 5.2 1.2;1.2 5.12 1.4;1 5 1.6;1.2 5.1 1.8];

start_node = 101;
end_node = length(beta_bin);
new_order = zeros(1,k);

for i=1:k
    temp_HMRF = zeros(n_dispara,2);
    temp_GMM = zeros(n_dispara,1);
    for j= 1:d
        temp_HMRF(j,1) = mean(mu_bin(i,j,start_node:end_node));
        temp_HMRF(j,2) = std(mu_bin(i,j,start_node:end_node));
        temp_GMM(j)= mu_bin(i,j,1);        
    end
    for p = 1:d
        for q = p:d
            temp_HMRF(d+(2*d-p+2)*(p-1)/2+(q-p+1),1) = mean(SIGMA_bin(p,q,i,start_node:end_node));
            temp_HMRF(d+(2*d-p+2)*(p-1)/2+(q-p+1),2) = std(SIGMA_bin(p,q,i,start_node:end_node));
            temp_GMM(d+(2*d-p+2)*(p-1)/2+(q-p+1),1) = SIGMA_bin(p,q,i,1);
        end
    end
    
    temp_mu = temp_HMRF(1:d,1)';
    compare_matrix = abs(ones(k,1)*temp_mu - MU);
    dist = sqrt(sum(compare_matrix.^2,2));
    [~,I] = min(dist);
    HMRF_matrix(((I-1)*n_dispara+1):((I-1)*n_dispara+n_dispara),:)=temp_HMRF;
    GMM_matrix(((I-1)*n_dispara+1):((I-1)*n_dispara+n_dispara))=temp_GMM;
    new_order(i) = I;
end

HMRF_matrix(k*n_dispara+1,1) = mean(beta_bin(start_node:end_node));
HMRF_matrix(k*n_dispara+1,2) = std(beta_bin(start_node:end_node));

MCR_HMRF=CalMCR(MID_list_true,MC_inferred(:,start_node:end_node),new_order);

HMRF_matrix(k*n_dispara+2,1) = mean(MCR_HMRF);
HMRF_matrix(k*n_dispara+2,2) = std(MCR_HMRF);

MCR_GMM=CalMCR(MID_list_true,MC_inferred(:,1),new_order);
GMM_matrix(k*n_dispara+1) = MCR_GMM;

end

