function [EMI_image,logEMI_image] = calibration(EMI_image,calib_para)
% calibrate and transfer EMI data y= k*x+b

addpath('./Inpaint_nans');

k = calib_para(1);
b = calib_para(2);
EMI_image = k*EMI_image+b;
m_nan = (isnan(EMI_image));
flag = nanmin(EMI_image(:));

if flag > 0
    fprintf('the min value is greater than 0: min(EMI) = %f \n',flag);
    fprintf('do log transfer');
    logEMI_image = log(EMI_image);    
else
    m_negtive = (EMI_image <= 0);
    n_negtive = sum(m_negtive(:));
    fprintf('the min value is smaller than 0: min(EMI) = %f; and the percentage of negtive value is: %f \n',flag,n_negtive/length(EMI_image(:))*100);
    fprintf('cannot do log transfer directly\n');
    EMI_image(m_negtive) = NaN;
    fprintf('apply script "inpaint_nans "\n');
    logEMI_image = log(EMI_image);    
    logEMI_image = inpaint_nans(logEMI_image,1);
    logEMI_image(m_nan) = NaN;
    EMI_image = exp(logEMI_image);
end