clearvars -except Element MC_simulated;
clc;
addpath('./Functions');
full_image = MC_simulated(:,200);

n = length(Element.Color);
k = 3000;
y = randsample(n,k);
MC_knowninfo = zeros(n,1);
for i = 1:k
    MC_knowninfo(y(i))=MC_simulated(y(i),200);
end
