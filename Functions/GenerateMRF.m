function [MC_bin,U_bin,beta_bin]=GenerateMRF(Element,MC_ini,Mset,Chain_length,beta_ini,SigmaProp_ini)
% This function is designed to perform adaptive MCMC simulation of the MRF and the beta vectors at the same time.

C=4.615;
n=length(MC_ini);
%===== config the scan oder ====
[para_scanorder,num_of_color] = chromaticClassification(Element);
%===== pre-allocation ==========
MC_bin = NaN(n,Chain_length);
U_bin = NaN(n,Chain_length - 1);
beta_bin(length(beta_ini),Chain_length) = 0;
%===== assign known info =======
MC_bin(:,1) = MC_ini;
beta_bin(:,1) = beta_ini;
%===== generate the chain ======

t0=100;
error=1e-5;
x = beta_bin(:,1)';
x_bar=x;
d = length(x);
sd=2.4^2/d;
SigmaProp=SigmaProp_ini;

for i=2:Chain_length
    x_barBefore=x_bar;
    if i<=100
        T=C/log(1+i);
    else
        T=C/log(101);
    end    
    [MC_bin(:,i),U_bin(:,i-1)]=Gibbs_samplling(Element,MC_bin(:,i-1),Mset,T,para_scanorder,num_of_color,beta_bin(:,i-1));    
    beta_bin(:,i) = updateBeta(Element,Mset,T,MC_bin(:,i),beta_bin(:,i-1),SigmaProp);
    x = beta_bin(:,i)';
    x_bar=(x_barBefore*(i-1)+x)/i;    
    if i>=t0+1
        SigmaProp = (i-1)/i*SigmaProp+sd/i*(i*(x_barBefore'*x_barBefore)-(i+1)*(x_bar'*x_bar)+(x'*x)+error*eye(d));        
    end
    display(i);
    display(beta_bin(:,i));
    display(SigmaProp);
end
end