function plotMixture(mu,sigma,MID_list)
% this function is used for plotting mixtures with a single pdf

num_of_cluster = size(mu,1);
p(num_of_cluster) = 0;
for i = 1:num_of_cluster
    p(i) = sum(MID_list==i);
end   
mixture = gmdistribution(mu,sigma,p);
ezcontour(@(x1,x2)pdf(mixture,[x1 x2]));
end