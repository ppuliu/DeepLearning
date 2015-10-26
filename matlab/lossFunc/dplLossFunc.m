function [cost, grad]=dplLossFunc(predict, target,lossFuncName)
%calculate the loss and their corresponding derivatives
%
% SYNOPSIS: [cost, grad]=dplLossFunc(predict, target)
%
% INPUT predict: predicted probabilities
%		target: target value              
%
% OUTPUT cost: scalar
%			grad: same dimension as predict  
%
% REMARKS
%
% created with MATLAB ver.: 8.3.0.532 (R2014a) on Mac OS X  Version: 10.9.5 Build: 13F34 
%
% created by: Honglei Liu
% DATE: 26-Oct-2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(strcmp(lossFuncName,'mse'))
    t=predict-target;
    cost=(t(:)'*t(:))/2;
    grad=t;
else if(strcmp(lossFuncName,'binary_crossentropy'))
       t=target.*log(predict)+(1-target).*log(1-predict);
       cost=-sum(t(:));
       grad=(predict-target)./(predict.*(1-predict));
    end
end

end