clc;clear;
addpath('./Functions');
%% define the block_size
block_size = 20;

%% construct Element
x = 1:block_size;
y = 1:block_size;
order = 1;
Element = constructElements(x',y',0,order);

%% generate toy data sets

latent_field(block_size^2,1) = 0;

mu=[6 9;4 12;9 14];
SIGMA(2,2,3)=0;
SIGMA(:,:,1)=3*[0.5625 0.225; 0.225 0.675];
SIGMA(:,:,2)=3*[1.125 0.225; 0.225 0.675];
SIGMA(:,:,3)=3*[0.5625 0.0225; 0.0225 0.675];

case_ID = 2;
% ========== case 1 =======================
if case_ID == 1
    for i = 1:block_size^2;
        d = norm(Element.Center(i,:)-[10,10,0]);
        if d <= 5
            latent_field(i) = 1;
        else if d>5 && d<=8
                latent_field(i) = 2;
            else
                latent_field(i) = 3;
            end
        end
    end
end
% ========== case 2 =====================
if case_ID == 2
    for i = 1:block_size^2;
        d = abs(Element.Center(i,2) - 5*sin(Element.Center(i,1)/20*2*pi)-10);
        if d <= 2
            latent_field(i) = 1;
        else if d > 2 && d <= 5
                latent_field(i) = 2;
            else
                latent_field(i) = 3;
            end
        end
    end
end
% ========================================

field_value = simulateSoftData(Element,latent_field,mu,SIGMA);

figure;
plotField(Element,latent_field);
title('latent field');

figure;
plotField(Element,field_value(:,1));
title('observed field 1');

figure;
plotField(Element,field_value(:,2));
title('observed field 2');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(mu,SIGMA,field_value,latent_field,labels);

%% segmentation
nbr_of_clusters = 3;
Chain_length = 200;
verbose = 1;
% =============================
seg = segmentation(Element,field_value,nbr_of_clusters,Chain_length,verbose);
% =============================

figure;
plotField(Element,seg.latent_field_est);
title('segmentation result');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(seg.MU_hat,seg.COV_hat,seg.field_value,seg.latent_field_est,labels);

figure;
plotField(Element,seg.InfEntropy);
title('InfEntropy');

%% chain diagonose 
iter = 2;

figure;
plotField(Element,seg.MC_inferred(:,iter));
title('GMM classification result');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(seg.mu_bin(:,:,iter),seg.SIGMA_bin(:,:,:,iter),seg.field_value,seg.MC_inferred(:,iter),labels);

% ======================
new_order = [1 2 3]; % manually adjust the order to make it compatible with the true latent field !!!!!!
% ======================
MCR=CalMCR(latent_field,seg.MC_inferred,new_order);
figure;
plot(2:Chain_length,MCR(2:Chain_length));
title('MCR');
xlabel('Iteration');
ylabel('MCR');
%% create movie
clc;clear;close all;
load toy_example_case1.mat;

clearvars F;

F(Chain_length-1) = struct('cdata',[],'colormap',[]);


for j = 1:(Chain_length-1)
    fig = figure('Position',[1 1 1920 470]);
    subplot(1,3,1);
    plotField(Element,seg.MC_inferred(:,j+1));
    colorbar off;
    subplot(1,3,2);
    labels = {'feature 1','feature 2'};
    mixturePlot(seg.mu_bin(:,:,j+1),seg.SIGMA_bin(:,:,:,j+1),seg.field_value,seg.MC_inferred(:,j+1),labels);
    subplot(1,3,3);
    plot(2:j+1,MCR(2:j+1));
    title('MCR');
    xlabel('Iteration');
    ylabel('MCR');
    drawnow;
    t_pause = 5;
    pause(t_pause);
    F(j) = getframe(fig);    
    close all;
    clearvars fig;
    display(j);
end

[h, w, p] = size(F(1).cdata);  % use 1st frame to get dimensions
hf = figure; 
% resize figure based on frame's w x h, and place at (150, 150)
set(hf, 'position', [150 150 w h]);
axis off

movie(hf,F);

% movie2avi(F,'segmentation.avi');
v = VideoWriter('segmentation','MPEG-4');
open(v);
writeVideo(v,F);
close(v);

