function [R,ux,uy] = mergeImage(S,X,Y)

nbr_image = length(S);
F = [];

for i = 1:nbr_image
    F = cat(1,F,formatField(X{i},Y{i},S{i}));
end

[ux,~,bx2] = unique(F(:,1));
[uy,~,by2] = unique(F(:,2));
R = accumarray( [max(by2)-(by2-1),bx2], F(:,3),[],@nanmean, NaN);

figure('units','normalized','outerposition',[0 0 1 1]);
imagescwithnan(ux,uy,R,viridis,[1 1 1]);
end