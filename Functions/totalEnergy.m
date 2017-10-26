function U = totalEnergy(Element,MC,beta)
ChainLength = size(MC,2);
U = zeros(ChainLength - 1,1);
N = Element.num_of_elements;
for k = 2:ChainLength
    MID_list=MC(:,k);
    U_temp=NaN(N,1);    
    parfor i=1:N
        if ~isnan(MC(i,k))
            U_temp(i) = 0;
            selfU = Element.SelfU(i,:);
            self_potential = selfU(MID_list(i));
            neighbor_list = Element.Neighbors{i};
            direction_list = Element.Direction{i};
            flag = MID_list(i)*ones(length(neighbor_list),1) ~= MID_list(neighbor_list);
            if size(beta,2) == 1
                neighbor_potential = sum(beta(direction_list(flag)));
            else
                neighbor_potential = sum(beta(direction_list(flag),k));
            end
            U_temp(i) = U_temp(i) + self_potential + neighbor_potential;
        end
    end
    U(k-1,1)=nansum(U_temp);
end
end
