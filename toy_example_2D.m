clc;clear;close all;
addpath('./Functions');
%% define the block_size
block_size = 40;

%% construct Element
x = 1:block_size;
y = 1:block_size;
order = 1;
Element = constructElements(x',y',0,order);

%% generate toy data sets

% initialize the latent field
latent_field(Element.num_of_elements,1) = 0;

% define the parameters of the feature space
mu=[6 9;4 12;9 14];
SIGMA(2,2,3)=0;
SIGMA(:,:,1)=3*[0.5625 0.225; 0.225 0.675];
SIGMA(:,:,2)=3*[1.125 0.225; 0.225 0.675];
SIGMA(:,:,3)=3*[0.5625 0.0225; 0.0225 0.675];

% ========== set which case to use ========
case_ID = 2;
% ========== case 1 =======================
if case_ID == 1
    for i = 1:block_size^2;
        d = norm(Element.Center(i,:)-[20,20,0]);
        if (d <= block_size/4)
            latent_field(i) = 1;
        else if (d > block_size/4) && (d <= block_size/2.5)
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
        d = abs(Element.Center(i,2) - (block_size/4)*sin(Element.Center(i,1)/block_size*2*pi)-block_size/2);
        if d <= block_size/10
            latent_field(i) = 1;
        else if d > block_size/10 && d <= block_size/4
                latent_field(i) = 2;
            else
                latent_field(i) = 3;
            end
        end
    end
end
% ========== plot the simulated data ====================
rng(6);
observed_features = simulateSoftData(Element,latent_field,mu,SIGMA);

figure;
plotField(Element,latent_field,jet);
title('latent field');

figure;
plotField(Element,observed_features(:,1),jet);
title('observed field 1');

figure;
plotField(Element,observed_features(:,2),jet);
title('observed field 2');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(mu,SIGMA,observed_features,latent_field,labels);

%% segmentation
num_of_clusters = 3;
Chain_length = 300;
dimension = 2;
beta_initial = [];
% =============================
seg = segmentation(Element,dimension,beta_initial,observed_features,num_of_clusters,Chain_length);
% =============================

figure;
plotField(Element,seg.latent_field_est,jet);
title('segmentation result');

figure;
plotField(Element,seg.InfEntropy,jet);
title('InfEntropy');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(seg.MU_hat,seg.COV_hat,seg.field_value,seg.latent_field_est,labels);

figure;
plot(1:length(seg.totalEnergy),seg.totalEnergy);
title('totalEnergy');
xlabel('Iteration');
ylabel('total energy');

%% chain diagonose 
iter = 30;

figure;
plotField(Element,seg.MC_inferred(:,iter),jet);
title('GMM classification result');

figure;
labels = {'feature 1','feature 2'};
mixturePlot(seg.mu_bin(:,:,iter),seg.SIGMA_bin(:,:,:,iter),seg.field_value,seg.MC_inferred(:,iter),labels);

% ======================
new_order = [2 1 3]; % manually adjust the order to make it compatible with the true latent field !!!!!!
% ======================
MCR=CalMCR(latent_field,seg.MC_inferred,new_order);
figure;
plot(2:Chain_length,MCR(2:Chain_length));
title('MCR');
xlabel('Iteration');
ylabel('MCR');

% =========================================================================
% %% create movie
% F(Chain_length-1) = struct('cdata',[],'colormap',[]);
% t_pause = 5;
% for j = 1:(Chain_length-1)
%     fig = figure('Position',[1 1 1000 1000]);
%     %============================================
%     subplot(2,2,1);
%     plotField(Element,seg.MC_inferred(:,j+1));
%     colorbar off;
%     %============================================
%     subplot(2,2,2);
%     labels = {'feature 1','feature 2'};
%     mixturePlot(seg.mu_bin(:,:,j+1),seg.SIGMA_bin(:,:,:,j+1),seg.field_value,seg.MC_inferred(:,j+1),labels);    
%     %==============================================
%     subplot(2,2,3);
%     plot(2:j+1,MCR(2:j+1));
%     title('MCR');
%     xlim([1 Chain_length]);
%     ylim([0 max(MCR)+0.02]);
%     xlabel('Iteration');
%     ylabel('MCR');
%     %==============================================
%     subplot(2,2,4);
%     plot(2:j+1,seg.beta_bin(:,2:j+1));
%     title('beta');
%     xlim([1 Chain_length]);
%     ylim([0 max(seg.beta_bin(:))+0.05]);
%     xlabel('Iteration');
%     ylabel('beta');    
%     %===============================================
%     drawnow;    
%     pause(t_pause);
%     F(j) = getframe(fig);    
%     close all;
%     clearvars fig;
%     display(j);
% end
% 
% [h, w, p] = size(F(1).cdata);  % use 1st frame to get dimensions
% hf = figure; 
% % resize figure based on frame's w x h, and place at (150, 150)
% set(hf, 'position', [150 150 w h]);
% axis off
% 
% movie(hf,F);
% 
% %% movie2avi(F,'segmentation.avi');
% v = VideoWriter('~/Videos/segmentation.avi','Motion JPEG AVI');
% open(v);
% writeVideo(v,F);
% close(v);