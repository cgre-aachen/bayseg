function Z_hat=OrdinaryKriging(S0,S,Z,CovarianceFunction,Arguments)
n_S=size(Z,1);
C=zeros(n_S,n_S);

%===calculate covariance matrix=====================================
for i=1:n_S
    for j=1:n_S
        C(i,j)=feval(CovarianceFunction,S(i,:),S(j,:),Arguments);
    end
end
%===================================================================        
n_S0=size(S0,1);
Z_hat=zeros(n_S0,3);
for k=1:n_S0
    c=zeros(n_S,1);
    for l=1:n_S
        c(l)=feval(CovarianceFunction,S0(k,:),S(l,:),Arguments);
    end
    Z_hat(k,:)=c'/C*Z+(1-c'/C*ones(n_S,1))*(ones(1,n_S)/C*ones(n_S,1))^(-1)*(ones(1,n_S)/C*Z);    
end

end