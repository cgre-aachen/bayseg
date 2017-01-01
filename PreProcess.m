function F = PreProcess(EMI_filename,loc_filename)
addpath('./Functions');
%load allData_ratio_july.mat July_ratio x_july y_july;
%load allData_F760_june.mat June_760 x_june y_june;
load allData_NDVI.mat NDVI x y;
%======================================================
EMI_data = csvread(EMI_filename,1,0);
loc = csvread(loc_filename,1,0);

% process EMI_image
[ux_EMI,~,bx] = unique(EMI_data(:,1));
[uy_EMI,~,by] = unique(EMI_data(:,2));
EMI_matrix = accumarray( [max(by)-(by-1),bx], EMI_data(:,3),[],[], NaN);

% process NDVI_image
[NDVI_matrix,ux_NDVI,uy_NDVI,~] = getField(NDVI,x-20,y+20,loc);
% F04 x-20 y+10
% F05 x-20 y+20

% generate unified grid
resolusion = 5;
ux = (min([ux_EMI;ux_NDVI]):resolusion:max([ux_EMI;ux_NDVI]))';
uy = (min([uy_EMI;uy_NDVI]):resolusion:max([uy_EMI;uy_NDVI]))';

% generate unified image
EMI_image = changeResolution(EMI_matrix,ux_EMI,uy_EMI,ux,uy);
NDVI_image = changeResolution(NDVI_matrix,ux_NDVI,uy_NDVI,ux,uy);


logEMI = log(EMI_image);
logEMI_z_score = (logEMI-nanmean(logEMI(:)))/nanstd(logEMI(:));
NDVI_z_score = (NDVI_image-nanmean(NDVI_image(:)))/nanstd(NDVI_image(:));

F.loc = loc;
F.EMI_image = EMI_image;
F.logEMI_z_score = logEMI_z_score;
F.NDVI_image = NDVI_image;
F.NDVI_z_score = NDVI_z_score;
F.ux1 = ux;
F.uy1 = uy;
end