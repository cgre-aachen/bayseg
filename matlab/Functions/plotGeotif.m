function plotGeotif(A,R,cmap,climits)
n_step = size(cmap,1);
temp = A;
temp(temp < climits(1)) = climits(1);
temp(temp > climits(2)) = climits(2);
norm_temp = (temp - climits(1))/(climits(2)-climits(1));
temp_idx = double(round(norm_temp*n_step));
mapshow(temp_idx,cmap,R);
caxis(climits);
colormap(cmap);
end