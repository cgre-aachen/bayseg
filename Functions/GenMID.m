function MID=GenMID(Mset,P,N)
% Mset : the set of Materials
% P : the list of probability of each material, must be column vector!
% N : the number of trials need to be generated.

k=length(P);
L=tril(ones(k,k),0);
CDF=[0;L*P];

MID=zeros(N,1);
r=rand(N,1);
for i=1:N
    temp=r(i)-CDF;
    idx=sum(temp>0);
    MID(i)=Mset(idx);
end

end