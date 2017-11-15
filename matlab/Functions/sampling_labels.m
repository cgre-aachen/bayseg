function [MID_list,U_list]=sampling_labels(Element,MID_list,Mset,T,para_scanorder,num_of_color,y,mu,SIGMA,beta_value)
% mu is k by d
% SIGMA is d by d by k

n_Mset=length(Mset);
U_list = NaN(length(MID_list),1);

for i=1:num_of_color    
    pointer=para_scanorder{i};
    n=sum(pointer>0);    
    temp_y = y(pointer,:);    
    temp_U = Element.SelfU(pointer,:);
    temp_Nei = Element.Neighbors(pointer);
    temp_Direc = Element.Direction(pointer);    
    temp_MID_list_old = MID_list(pointer);    
    temp_MID_list_new = temp_MID_list_old;
    temp_U_list = U_list(pointer);
    parfor idx = 1:n
        if ~isnan(temp_MID_list_old(idx))
            U=temp_U(idx,:); % assign the energy of sigle site clique
            %========= calculate the likelihood energy ==============            
            for k=1:n_Mset
                C=SIGMA(:,:,k);
                Uy=0.5*(temp_y(idx,:)-mu(k,:))/C*(temp_y(idx,:)-mu(k,:))'+0.5*log(det(C));
                U(k)=U(k)+Uy; % add the likelihood energy
            end
            %========= calculate the MRF energy =====================
            current_nei_list = temp_Nei{idx};
            current_direction = temp_Direc{idx};
            n_neighbor=length(current_nei_list);            
            M_center = ones(n_neighbor,1)*Mset;            
            M_neighbor = MID_list(current_nei_list)*ones(1,n_Mset);            
            is_zero = (M_center==M_neighbor) | isnan(M_neighbor);           
            M_beta = beta_value(current_direction)*ones(1,n_Mset); % beta_value is n_direction-by-1 vector.           
            M_beta(is_zero) = 0;            
            U = U + sum(M_beta);
            %====calculate the probability of each label===============================
            P=exp(-U/T)/sum(exp(-U/T));
            %==========================================================================
            temp_MID_list_new(idx)=GenMID(Mset,P',1);
            if temp_MID_list_old(idx) > 0
                temp_U_list(idx) = U(temp_MID_list_old(idx));
            end
        end        
    end    
    MID_list(pointer)=temp_MID_list_new;
    U_list(pointer) = temp_U_list;
end
end