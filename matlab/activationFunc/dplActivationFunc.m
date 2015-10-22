function [val, grad] = dplActivationFunc(input, method)
%DPLACTIVATIONFUNC implements various activation functions
%
% SYNOPSIS: [val, grad] = dplActivationFunc(input, method='sigmoid')
%
% INPUT input : tensor
%		method : choice of activation functions
%		
%			'sigmoid'
%			'tanh'                                     
%
% OUTPUT val : value after applying activation function
%        grad : derivatives 
%
% REMARKS
%
% created with MATLAB ver.: 8.0.0.783 (R2012b) on Microsoft Windows 7 Version 6.1 (Build 7601: Service Pack 1)
%
% created by: Honglei Liu
% DATE: 14-Oct-2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(strcmp(method,'sigmoid'))
    val=1./(1+exp(-input));
    grad=val.*(1-val);
else if(strcmp(method,'tanh'))
        val=tanh(input);
        grad=1-val.^2;
    else
        val=null;
        grad=null;
    end
end
