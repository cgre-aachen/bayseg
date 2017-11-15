clc;clear;close all;
addpath('./Functions');
x = (1:128)';
y = (1:128)';
order = 1;
Element = constructElements(x,y,0,order);
%%
Mset=[1 2 3];
Chain_length = 300;
beta = [2 -0.5 2 -0.5]'; % beta is a column vector
MC_ini = zeros(length(x)*length(y),1);
Element = FixElement(Element,MC_ini); % if 0 is filled at a given pixel, the label is not fixed.
Element = CalculateU(Element,zeros(1,length(Mset)));
Element = detectNeighborDirection(Element,2);
[MC_simulated,U_bin] = SimulateMRF(Element,MC_ini,Mset,Chain_length,beta);

figure;
plotField(Element,MC_simulated(:,Chain_length),jet);
%%
U_chain = nansum(U_bin);
figure;
plot(2:length(U_chain),U_chain(2:end));