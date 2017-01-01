clearvars -except Element MC_simulated MC_knowninfo full_image;
clc;
addpath('./Functions');

x = (1:128)';
y = (1:128)';
order = 1;
% Element = constructElements(x,y,0,order);

Mset=[1 2 3];
Chain_length=200;
n_chain = 1;


beta_ini = mvnrnd(zeros(1,4),diag(ones(4,1)))'; % beta is a column vector
SigmaProp_ini = diag(0.01*ones(4,1));
MC_ini = MC_knowninfo;
plotField(Element,MC_ini);

Element = FixElement(Element,MC_ini);
Element = CalculateU(Element,zeros(1,length(Mset)));
Element = detectNeighborDirection(Element,2);

for i = 1:n_chain
    [MC_est,beta_bin]=GenerateMRF(Element,MC_ini,Mset,Chain_length,beta_ini,SigmaProp_ini);    
end

plotField(Element,MC_est(:,Chain_length));

%MCR=CalMCR(MC(:,500),MC_est,[1 2 3]);

% load MRFtruth_MRF3.mat MID_list_true
%load dataStructureFine_50x50x25.mat Element
% mu=[6 9;4 12;7 14];
% SIGMA(2,2,3)=0;
% SIGMA(:,:,1)=[0.5625 0.225;0.225 0.675];
% SIGMA(:,:,2)=[1.125 0.225;0.225 0.675];
% SIGMA(:,:,3)=[0.5625 0.0225;0.0225 0.675];
% 
% for i = 1:size(MID_list_true,2)
%     y=simulateSoftData(Element,MID_list_true(:,i),mu,SIGMA);
%     save(['softData_MRFcase_0.5xSIGMA_',num2str(i),'.mat'],'y');
%     %WriteGmsh_y('myscriptFine.msh','MRFcase_y_2.5xSIGMA_MRF3.msh',y,Element);
% end(length(x)*length(y),1)(length(x)*length(y),1)(length(x)*length(y),1)