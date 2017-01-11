%% define source data

clc;clear;close all;
addpath('./Functions');
<<<<<<< HEAD
EMI_FileList = {'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_ME_V32_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_ME_V71_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_ME_V118_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H35_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H49_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H71_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H97_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H135_1m.csv';
                'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H180_1m.csv';};
loc_filename = 'C:/users/wang/Documents/MATLAB/testing_data/Schophoven_SE_H180_loc.csv';

calib_para = [1.11 16.5;
              1.03 7.2;
              1.16 -0.5;
              1.24 3.1;
              1.40 1.3;
              1.72 1.8;
              1.74 -5.1;
              1.51 -3.4;
              1.4 -3.0];


%% prefprocess
F01 = PreProcess(EMI_FileList,loc_filename,calib_para);

F01.weighted_EMI = weightedAvg(F01.EMI_image);
log_weighted_EMI = log(F01.weighted_EMI);
F01.log_weighted_EMI_z_score = (log_weighted_EMI-nanmean(log_weighted_EMI(:)))/nanstd(log_weighted_EMI(:));
=======
EMI_FileList = {'./F01_ME_H118_1m.csv'};
loc_filename = './F01_ME_H118_loc.csv';

%% preprocess
F01 = PreProcess(EMI_FileList,loc_filename);
>>>>>>> 4d57210847952564ecf22b6cc437e0ce9ddc4a42

figure;
imagescwithnan(F01.ux1,F01.uy1,F01.EMI_image,mycmap,[1 1 1]);
title('EMI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(F01.ux1,F01.uy1,F01.NDVI_image,viridis,[1 1 1]);
title('NDVI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;
%%
F = {F01};
%[gl_Element,gl_field_value,F] = Global_preprocess(F);
[~,~,F] = Global_preprocess(F);

%% segmentation
dimension = 2;
beta_initial = []; % do not specify initial value (i.e. randomly generate intial value)
num_of_clusters = 2;
Chain_length = 50;
% =============================
seg = segmentation(F{1}.Element,dimension,beta_initial,F{1}.field_value,num_of_clusters,Chain_length);
% =============================
%%
Ext_Chain_length = 100;
seg = ExtendChain_para(seg,Ext_Chain_length);
% =============================
%% postprocess
figure;
plotField(F{1}.Element,seg.latent_field_est);
title('segmentation result');

figure;
plotField(F{1}.Element,seg.InfEntropy);
title('Information Entropy');

figure;
labels = {'NDVI z-score','log(EMI) z-score'};
mixturePlot(seg.MU_hat,seg.COV_hat,seg.field_value,seg.latent_field_est,labels);

figure;
imagescwithnan(F{1}.ux1,F{1}.uy1,F{1}.NDVI_z_score,viridis,[1 1 1]);
title('NDVI image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(F{1}.ux1,F{1}.uy1,F{1}.logEMI_z_score,mycmap,[1 1 1]);
title('log(EMI) image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

% save F01_ME_H118_results.mat F F01 gl_Element gl_field_value seg