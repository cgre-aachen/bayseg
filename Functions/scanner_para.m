function [MID_list,mu,SIGMA,beta]=scanner_para(Element,MID_list,Mset,T,para_scanorder,num_of_color,y,mu,SIGMA,beta,prior_para)
% mu is k by d
% SIGMA is d by d by k
tic;
MID_list=sampling_labels(Element,MID_list,Mset,T,para_scanorder,num_of_color,y,mu,SIGMA,beta);
[mu,SIGMA,beta]=sampling_gmdisPara2(Element,MID_list,Mset,T,y,mu,SIGMA,beta,prior_para);
toc;
end