function [InfEntropy,TotalInfEntr]=PostprocessInfoEntr(Prob,Mset)
N=size(Prob,1);
n=length(Mset);
InfEntropy=NaN(N,1);

parfor i=1:N
    if ~isnan(sum(Prob(i,:)))
        InfEntropy(i) = 0;
        for j=1:n
            if Prob(i,j)~=0
                InfEntropy(i)=InfEntropy(i)+sum(Prob(i,j)*log(Prob(i,j)));
            end
        end
    end      
end
InfEntropy=-InfEntropy;
TotalInfEntr=nansum(InfEntropy)/sum(~isnan(InfEntropy));