function [sample_mu1,sample_mu2,sample_SIGMA1,sample_SIGMA2,sample_SIGMA12]=plotChain(mu_bin,SIGMA_bin,k)
chain_length=size(mu_bin,3);
sample_mu1(chain_length)=0;
sample_mu2(chain_length)=0;
sample_SIGMA1(chain_length)=0;
sample_SIGMA2(chain_length)=0;
sample_SIGMA12(chain_length)=0;
for i=1:chain_length
    sample_mu1(i)=mu_bin(k,1,i);
    sample_mu2(i)=mu_bin(k,2,i);
    sample_SIGMA1(i)=SIGMA_bin(1,1,k,i);
    sample_SIGMA2(i)=SIGMA_bin(2,2,k,i);
    sample_SIGMA12(i)=SIGMA_bin(1,2,k,i);
end
figure;
plot(1:chain_length,sample_mu1(1:chain_length),'r');
figure;
plot(1:chain_length,sample_mu2(1:chain_length),'r');
figure;
plot(1:chain_length,sample_SIGMA1,'b');
figure;
plot(1:chain_length,sample_SIGMA2,'b');
figure;
plot(1:chain_length,sample_SIGMA12,'b');
end