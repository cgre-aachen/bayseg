%% add the function path
clc;clear;close all;
addpath('./Functions');

%%
load 2D_sample_data.mat

figure;
imagescwithnan(F.ux,F.uy,F.EMI_image,mycmap);
title('EMI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(F.ux,F.uy,F.NDVI_image,viridis);
title('NDVI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;
%%
order = 1;
F.Element = constructElements(F.ux,F.uy,0,order);
F.field_value = cat(2,retrieve(F.Element,F.NDVI_image),retrieve(F.Element,F.EMI_image));
F.field_value(isnan(sum(F.field_value,2)),:) = NaN;

%% segmentation
dimension = 2;
beta_initial = []; % do not specify initial value (i.e. randomly generate intial value)
num_of_clusters = 2;
Chain_length = 50;
% =============================
seg = segmentation(F.Element,dimension,beta_initial,F.field_value,num_of_clusters,Chain_length);
% =============================
%% extend the Markov Chain
Ext_Chain_length = 50;
seg = ExtendChain_para(seg,Ext_Chain_length);
% =============================
%% postprocess
figure;
plotField(F.Element,seg.latent_field_est,viridis);
title('segmentation result');

figure;
plotField(F.Element,seg.InfEntropy,jet);
title('Information Entropy');

figure;
labels = {'NDVI','EC_a'};
mixturePlot(seg.MU_hat,seg.COV_hat,seg.field_value,seg.latent_field_est,labels);

figure;
imagescwithnan(F.ux,F.uy,F.NDVI_image,viridis);
title('NDVI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(F.ux,F.uy,F.EMI_image,mycmap);
title('EMI image');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;