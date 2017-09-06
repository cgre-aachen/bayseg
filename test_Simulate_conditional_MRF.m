clc;clear;close all;
addpath('./Functions');

%% define the block_size
block_size = 40;

%% construct Element and do simulation
x = 1:block_size;
y = 1:block_size;
order = 1;
Element = constructElements(x',y',0,order);

dimension = 2;
Mset=[1 2];
beta = [2 -0.5 2 -0.5]'; % beta is a column vector
num_of_elements = Element.num_of_elements;
num_of_known_pixels = round(0.05*num_of_elements);
sample = GenMID(Mset,[0.3;0.7],num_of_known_pixels);
sample_idx = randsample(num_of_elements,num_of_known_pixels);
MC_knowninfo = zeros(length(Element.Color),1); 
MC_knowninfo(sample_idx) = sample;

figure;
temp = MC_knowninfo;
temp(temp == 0) = NaN;
plotField(Element,temp,jet);
title('known info');
%%
Chain_length=500;

Element = FixElement(Element,MC_knowninfo);
Element = CalculateU(Element,[0,0]);
Element = detectNeighborDirection(Element,dimension);

MC_simulated=SimulateMRF(Element,MC_knowninfo,Mset,Chain_length,beta);

figure;
plotField(Element,MC_simulated(:,Chain_length),jet);
title('simulation result');

%%
U = totalEnergy(Element,MC_simulated,beta);
figure;
plot(2:Chain_length,U(1:Chain_length-1));