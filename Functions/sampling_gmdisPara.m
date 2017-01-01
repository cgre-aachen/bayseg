function [mu,SIGMA]=sampling_gmdisPara(Element,MID_list,Mset,T,para_scanorder,num_of_color,y,mu,SIGMA)
% mu is n_Mset by d
% SIGMA is d by d by k
%===== constract proposal function ===============
n_Mset=size(mu,1);
d=size(mu,2);

mu_star=zeros(n_Mset,d);
SIGMA_star=zeros(d,d,n_Mset);

for i=1:n_Mset
    jump_mu=mvnrnd(zeros(1,d),diag(0.0005*ones(1,d)),1);
    mu_star(i,:)=mu(i,:)+jump_mu;
    
    [V,D]=eig(SIGMA(:,:,i));
    jump_D=mvnrnd(zeros(1,d),diag(0.0001*ones(1,d)),1);
    SIGMA_star(:,:,i)=V*(D+diag(jump_D))*V';
end

%===== construct likihood function=================

p_y_old=zeros(length(Element),1);
p_y_star=zeros(length(Element),1);
for i=1:num_of_color
    
    box=para_scanorder{i};
    n=length(box);    
    LL2=zeros(n,n_Mset);
    LL3=zeros(n,1);
    LL4=cell(n,1);
    
    for ii=1:n
        address=box(ii);
        LL2(ii,:)=Element(address).SelfU;
        LL3(ii)=MID_list(address);        
        LL4{ii}=Element(address).Neighbors;
    end
    
    temp_p_y_old=zeros(n,1);
    temp_p_y_star=zeros(n,1);
    
    parfor idx=1:n
        pointer=box(idx);
        U=(LL2(idx,:)); % assign the energy of sigle site clique
        n_neighbor=length(LL4{idx});
        for iii=1:n_Mset
            for j=1:n_neighbor                
                MID_CurrentNeighbor=MID_list(LL4{idx}(j).Address);
                if MID_CurrentNeighbor~=0
                    if (MID_CurrentNeighbor-Mset(iii))~=0
                        U(iii)=U(iii)+0.8;
                        %U(iii)=U(iii)+LL4{idx}(j).Beta;%*abs(MID_CurrentNeighbor-Mset(k)); % add the energy from interaction
                    end
                end
            end            
        end
        P=exp(-U/T)/sum(exp(-U/T));
        f_y_old=zeros(1,n_Mset);
        f_y_star=zeros(1,n_Mset);
        for l=1:n_Mset
            f_y_old(l)=mvnpdf(y(pointer,:),mu(l,:),SIGMA(:,:,l));
            f_y_star(l)=mvnpdf(y(pointer,:),mu_star(l,:),SIGMA_star(:,:,l));
        end
        temp_p_y_old(idx)=P*f_y_old';
        temp_p_y_star(idx)=P*f_y_star';
    end
    
    for jj=1:n
        addr=box(jj);
        p_y_old(addr)=temp_p_y_old(jj);
        p_y_star(addr)=temp_p_y_star(jj);
    end   
end

%==========================================================================
 
logLikelihood_old=sum(log(p_y_old));
logLikelihood_star=sum(log(p_y_star));

if logLikelihood_star>logLikelihood_old
    mu=mu_star;
    SIGMA=SIGMA_star;
else
    accceptRate=exp(logLikelihood_star-logLikelihood_old);
    r=rand(1,1);
    if r<=accceptRate
        mu=mu_star;
        SIGMA=SIGMA_star;
    end
end

end