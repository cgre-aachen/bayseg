function Prob=PostprocessProbability(MC_inferred,Mset,startpoint)
N=size(MC_inferred,1); % number of elements
n_Mset=length(Mset);
ChainLength=size(MC_inferred,2);
Num_of_samples=ChainLength-startpoint+1;
Samples=MC_inferred(:,startpoint:ChainLength);

Prob=zeros(N,n_Mset);

parfor i=1:N
    for j=1:n_Mset
       Prob(i,j)=sum(Samples(i,:)==Mset(j))/Num_of_samples;       
    end
end

idx = sum(Prob,2) == 0;
Prob(idx,:) = NaN;