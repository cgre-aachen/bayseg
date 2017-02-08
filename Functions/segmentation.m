function seg = segmentation(Element,dimension,beta_initial,field_value,num_of_clusters,Chain_length)
% this is function is an implementation of the HMRF classification.
% verbose is a logical value 1: output the detailed results
%                            0: donot output the detailed results

% =============================

Mset= 1:num_of_clusters;

if ~isfield(Element,'SelfU')
    Element = CalculateU(Element,zeros(1,length(Mset)));
end

if ~isfield(Element,'Direction')
    Element = detectNeighborDirection(Element,dimension);
end

[para_scanorder,num_of_color]=chromaticClassification(Element);

if isempty(beta_initial)
    if dimension == 1
        num_of_diff_beta = 1;
    end
    if dimension == 2
        num_of_diff_beta = 4;
    end
    if dimension == 3
        num_of_diff_beta = 13;
    end
    beta_initial = mvnrnd(zeros(1,num_of_diff_beta),0.01*diag(ones(num_of_diff_beta,1)))'; % beta is a column vector! 3D case beta is 13x1, 2D case beta is 4x1
end

% =========== HMRF sampling ==================

[MC_inferred,mu_bin,SIGMA_bin,beta_bin]=GenChain_para(Element,Mset,Chain_length,para_scanorder,num_of_color,field_value,beta_initial);

% =========== Post_process ==================
startPoint = round(Chain_length/5);
[Prob,InfEntropy,TotalInfEntr]=PostEntropy(MC_inferred,Mset,startPoint);
[~,latent_field_est] = max(Prob,[],2);
latent_field_est(isnan(sum(Prob,2))) = NaN;
[MU_hat,MU_std] = CalCenter(mu_bin,startPoint);
[R,stds,COV_hat,COV_std] = CalCorrCoeff(SIGMA_bin,startPoint);
[beta_hat,beta_std] = CalBeta(beta_bin,startPoint);

% ========== output ==========================
seg.latent_field_est = latent_field_est;
seg.InfEntropy = InfEntropy;
seg.MU_hat = MU_hat;
seg.COV_hat = COV_hat;
seg.beta_hat = beta_hat;
seg.num_of_clusters = num_of_clusters;
seg.para_scanorder = para_scanorder;
seg.num_of_color = num_of_color;
seg.field_value = field_value;
seg.Element = Element;
seg.MC_inferred = MC_inferred;
seg.mu_bin = mu_bin;
seg.SIGMA_bin = SIGMA_bin;
seg.beta_bin = beta_bin;
seg.Prob = Prob;    
seg.TotalInfEntr = TotalInfEntr;
seg.MU_std = MU_std;
seg.COV_std = COV_std;
seg.beta_std = beta_std;
seg.CorrCoeffMatrix = R;
seg.stdMatrix = stds;
end