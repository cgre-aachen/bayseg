function M_weighted = weightedAvg(M)

sensitivity_depth = [0.75*[32 71 118] 1.5*[35 49 71 97 135 180]];
w = sensitivity_depth/sum(sensitivity_depth);

M_weighted = 0;

for i = 1:9
    temp = w(i)*M(:,:,i);
    M_weighted = M_weighted + temp;
end
end