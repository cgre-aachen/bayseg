function U=totalEnergy(Element,MC)
ChainLength=size(MC,2);
U=zeros(ChainLength,1);
N=length(Element);
for k=1:ChainLength
    MID_list=MC(:,k);
    U_temp=0;
    U_p=0;
    for i=1:N
        Potential=Element(i).SelfU;
        U_p=U_p+Potential(MID_list(i));
        n=length(Element(i).Neighbors);
        for j=1:n
            CurrentNeighborAddress=Element(i).Neighbors(j).Address;
            if MID_list(i)~=MID_list(CurrentNeighborAddress)
                U_temp=U_temp+Element(i).Neighbors(j).Beta;
            end
        end
    end
    U(k,1)=1/2*U_temp+U_p;
end
figure;
plot(1:ChainLength,U(1:ChainLength));
end
