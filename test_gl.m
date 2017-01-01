[gl_Element,gl_field_value,F] = Global_preprocess({f42,f43});

gl_seg = segmentation(gl_Element,gl_field_value,2,100,1);

plotField(F{1}.Element,gl_seg.latent_field_est(1:11448));
title('segmentation result');
plotField(F{2}.Element,gl_seg.latent_field_est(11449:22896));
title('segmentation result');

labels = {'NDVI z-score','log(EMI) z-score'};
mixturePlot(gl_seg.MU_hat,gl_seg.COV_hat,gl_seg.field_value,gl_seg.latent_field_est,labels);

% plot single field
figure;
imagescwithnan(f42.ux1,f42.uy1,f42.NDVI_z_score,viridis,[1 1 1]);
title('NDVI image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(f42.ux1,f42.uy1,f42.logEMI_z_score,mycmap,[1 1 1]);
title('log(EMI) image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

% plot single field
figure;
imagescwithnan(f43.ux1,f43.uy1,f43.NDVI_z_score,viridis,[1 1 1]);
title('NDVI image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;

figure;
imagescwithnan(f43.ux1,f43.uy1,f43.logEMI_z_score,mycmap,[1 1 1]);
title('log(EMI) image z-score');
xlabel('UTM-E [m]');
ylabel('UTM-N [m]');
axis equal;
