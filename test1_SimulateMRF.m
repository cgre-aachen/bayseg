% clear;
addpath('./HMRF_3D');
load squre_Element.mat Element
Mset=[1 2];
Chain_length=500;
n_chain = 1;
beta = [1.92 0.44 2.01 -0.46]'; % beta is a column vector
MC_knowninfo = zeros(length(Element),1);
Element=FixElement(Element,MC_knowninfo);
Element=CalculateU(Element,[0,0]);
Element = detectNeighborDirection(Element);
for i = 1:n_chain
    MC_simulated=SimulateMRF(Element,MC_knowninfo,Mset,Chain_length,beta);
end
%MCR=CalMCR(MC(:,450),MC_simulated,[1 2 3]);
%[Prob,InfEntropy,TotalInfEntr]=PostEntropy(MC_simulated,Mset,250);
save('MRF_simulated1.mat','MC_simulated');
WriteGmsh('squre128x128.msh','MRF_simulated1.msh',MC_simulated,0,0,Mset,Element,20,[1 1 0 0]);