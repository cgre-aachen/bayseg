function U = totalEnergy(Element,MC,beta)
ChainLength = size(MC,2);
U = zeros(ChainLength - 1,1);
N = Element.num_of_elements;
for k = 2:ChainLength
    MID_list=MC(:,k);
    U_temp=0;    
    for i=1:N
        selfU = Element.SelfU(i,:);
        self_potential = selfU(MID_list(i));        
        neighbor_list = Element.Neighbors{i};
        direction_list = Element.Direction{i};
        flag = MID_list(i)*ones(length(neighbor_list),1) ~= MID_list(neighbor_list);
        neighbor_potential = sum(beta(direction_list(flag)));
        U_temp = U_temp + self_potential + neighbor_potential;
    end
    U(k-1,1)=U_temp;
end
end
