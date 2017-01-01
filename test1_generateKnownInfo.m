addpath('./Functions');
clearvars -except Element;
clc;
x = (1:128)';
y = (1:128)';
order = 1;
% Element = constructElements(x,y,0,order);

Mset=[1 2 3];
Chain_length=200;
n_chain = 1;
beta = [2 -0.5 2 -0.5]'; % beta is a column vector
MC_ini = zeros(length(x)*length(y),1);
Element = FixElement(Element,MC_ini);
Element = CalculateU(Element,zeros(1,length(Mset)));
Element = detectNeighborDirection(Element,2);
for i = 1:n_chain
    MC_simulated = SimulateMRF(Element,MC_ini,Mset,Chain_length,beta);
end

% save('MRF_simulated_test.mat','MC_simulated');
plotField(Element,MC_simulated(:,Chain_length));

%WriteGmsh('cube32x32x32.msh','MRF_simulated_test_cube_1stOrider.msh',MC_simulated,0,0,Mset,Element,20,[1 1 0 0]);


% load MRF_simulated_test.mat MC_simulated
% 
% MID_list_true = MC_simulated(:,1000);
% mu=[6 9;4 12;7 14];
% SIGMA(2,2,3)=0;
% SIGMA(:,:,1)=4*[0.5625 0.225; 0.225 0.675];
% SIGMA(:,:,2)=4*[1.125 0.225; 0.225 0.675];
% SIGMA(:,:,3)=4*[0.5625 0.0225; 0.0225 0.675];
%  
% for i = 1:size(MID_list_true,2)
%     y=simulateSoftData(Element,MID_list_true(:,i),mu,SIGMA);
%     save(['softData_MRF2D_',num2str(i),'.mat'],'y');
%     WriteGmsh_y('squre128x128.msh',['softData_MRF2D_',num2str(i),'.msh'],y,Element);
% end