function F = PreProcess(EMI_FileList,loc_filename,calib_para)
addpath('./Functions');
%load allData_ratio_july.mat July_ratio x_july y_july;
%load allData_F760_june.mat June_760 x_june y_june;
load allData_NDVI.mat NDVI x y;
%======================================================
loc = csvread(loc_filename,1,0);

% process NDVI_image
[NDVI_matrix,ux_NDVI,uy_NDVI,~] = getField(NDVI,x-20,y+20,loc);
% F04 x-20 y+10
% F05 x-20 y+20
%========================================================

resolusion = 5;
num_EMI_files = length(EMI_FileList);
EMI_image = [];
logEMI_z_score = [];
for i = 1:num_EMI_files
    if i == 1
        EMI_data = csvread(EMI_FileList{1},1,0);
        [ux_EMI,~,bx] = unique(EMI_data(:,1));
        [uy_EMI,~,by] = unique(EMI_data(:,2));
        ux = (min([ux_EMI;ux_NDVI]):resolusion:max([ux_EMI;ux_NDVI]))';
        uy = (min([uy_EMI;uy_NDVI]):resolusion:max([uy_EMI;uy_NDVI]))';
        temp = accumarray( [max(by)-(by-1),bx], EMI_data(:,3),[],[], NaN);        
        EMI_image = changeResolution(temp,ux_EMI,uy_EMI,ux,uy);        
        [EMI_image,logEMI] = calibration(EMI_image,calib_para(i,:));
        logEMI_z_score = (logEMI-nanmean(logEMI(:)))/nanstd(logEMI(:));
    else
        EMI_data = csvread(EMI_FileList{i},1,0);
        [ux_EMI,~,bx] = unique(EMI_data(:,1));
        [uy_EMI,~,by] = unique(EMI_data(:,2));
        temp1 = accumarray( [max(by)-(by-1),bx], EMI_data(:,3),[],[], NaN);        
        temp2 = changeResolution(temp1,ux_EMI,uy_EMI,ux,uy);        
        [temp2,temp3] = calibration(temp2,calib_para(i,:));
        EMI_image = cat(3, EMI_image, temp2);
        temp4 = (temp3-nanmean(temp3(:)))/nanstd(temp3(:));
        logEMI_z_score = cat(3, logEMI_z_score, temp4);
    end
end
    
NDVI_image = changeResolution(NDVI_matrix,ux_NDVI,uy_NDVI,ux,uy);

NDVI_z_score = (NDVI_image-nanmean(NDVI_image(:)))/nanstd(NDVI_image(:));

F.loc = loc;
F.EMI_image = EMI_image;
F.logEMI_z_score = logEMI_z_score;
F.NDVI_image = NDVI_image;
F.NDVI_z_score = NDVI_z_score;
F.ux1 = ux;
F.uy1 = uy;
end