clc;
addpath('./Functions');
load north_berg_all.mat;
load loc_north_berg.mat;

[EMI_image_3,ux_3,uy_3,~] = getField(EMI_image,ux1,uy1,loc_3);
[LAI_image_3,~,~,~] = getField(LAI_image,ux1,uy1,loc_3);
[HS_image_3,~,~,~] = getField(HS_image,ux1,uy1,loc_3);
% ======================
HS_matrix = HS_image_3;
LAI_matrix = log(LAI_image_3);
EMI_matrix = log(EMI_image_3);
[HS_matrix,LAI_matrix,EMI_matrix] = matrixAlignment(ones(size(HS_matrix)),LAI_matrix,EMI_matrix);
% =============================
F = {LAI_matrix,EMI_matrix};
unified_x = ux_3;
unified_y = uy_3;
nbr_of_clusters = 2;
Chain_length = 100;
% =============================
seg3 = segmentation(F,unified_x,unified_y,nbr_of_clusters,Chain_length);
