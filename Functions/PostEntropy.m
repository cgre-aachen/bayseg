function [Prob,InfEntropy,TotalInfEntr]=PostEntropy(MC_bin,Mset,startpoint)
Prob=PostprocessProbability(MC_bin,Mset,startpoint);
[InfEntropy,TotalInfEntr]=PostprocessInfoEntr(Prob,Mset);
end