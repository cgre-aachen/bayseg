%% define source data
clc;clear;close all;
addpath('./Functions');
EMI_filename = '~/testing_data/Schophoven_SE_H180_1m.csv';
loc_filename = '~/testing_data/Schophoven_SE_H180_loc.csv';

% EMI_filename = 'F01_ME_H118_1m.csv';
% loc_filename = 'F01_ME_H118_loc.csv';
%% prefprocess
F01 = PreProcess(EMI_filename,loc_filename);

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
Chain_length = 100;
% =============================
seg = segmentation(F{1}.Element,dimension,beta_initial,F{1}.field_value,num_of_clusters,Chain_length);
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