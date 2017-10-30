function [MC_simulated,U_bin]=SimulateMRF(Element,MC_ini,Mset,Chain_length,beta)
C=4.615;
n=length(MC_ini);
%===== config the scan oder ====
[para_scanorder,num_of_color]=chromaticClassification(Element);
%===== pre-allocation ==========
MC_simulated = NaN(n,Chain_length);
U_bin = NaN(n,Chain_length - 1);
%===== assign known info =======
MC_simulated(:,1) = MC_ini;
%===== generate the chain ======
for i=2:Chain_length
    if i<=100
        T=C/log(1+i);
    else
        T=C/log(101);
    end
    
    [MC_simulated(:,i),U_bin(:,i-1)]=Gibbs_samplling(Element,MC_simulated(:,i-1),Mset,T,para_scanorder,num_of_color,beta);    
    
    if i == 2
        fprintf('Iteration = %4i',i-1);
    else        
        backspace = repmat('\b',1,4);       
        if i < Chain_length
            fprintf([backspace '%4i'],i-1);
        else
            fprintf([backspace '%4i \n'],i-1);
        end
    end        
end
end