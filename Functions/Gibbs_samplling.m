function [MID_list,U_list]=Gibbs_samplling(Element,MID_list,Mset,T,para_scanorder,num_of_color,beta)
n_Mset=length(Mset);
U_list = NaN(length(MID_list),1);

for i=1:num_of_color
    pointer=para_scanorder{i};
    n=sum(pointer>0);
    temp_fixed = Element.Fixed(pointer);
    temp_U = Element.SelfU(pointer,:);
    temp_U_list = U_list(pointer);
    temp_Nei = Element.Neighbors(pointer);
    temp_Direc = Element.Direction(pointer);    
    temp_MID_list_old = MID_list(pointer);    
    temp_MID_list_new = temp_MID_list_old; 
    parfor idx=1:n
        if ~isnan(MID_list(idx))
            U = temp_U(idx,:); % assign the energy of sigle site clique
            n_neighbor = length(temp_Nei{idx});
            M_center = ones(n_neighbor,1)*Mset;
            M_neighbor = MID_list(temp_Nei{idx})*ones(1,n_Mset);
            is_zero = (M_center==M_neighbor);
            is_zero(M_neighbor==0) = 1;
            M_beta = beta(temp_Direc{idx})*ones(1,n_Mset); % beta is n_direction-by-1 vector.
            M_beta(is_zero) = 0;
            U = U + sum(M_beta);
            %====calculate the probability of each label ==============================
            P=exp(-U/T)/sum(exp(-U/T));
            %==========================================================================
            if ~temp_fixed(idx)
                temp_MID_list_new(idx)=GenMID(Mset,P',1); %generate a candidate
            end
            if temp_MID_list_old(idx) > 0
                temp_U_list(idx) = U(temp_MID_list_old(idx));
            end
        end
    end
    MID_list(pointer) = temp_MID_list_new;
    U_list(pointer) = temp_U_list;
end
end