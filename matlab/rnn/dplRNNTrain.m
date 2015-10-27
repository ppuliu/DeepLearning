function [cost,optpara]=dplRNNTrain(nin, nh, nout, input, output, reg, lambda)
%train a RNN
%
% SYNOPSIS: [cost,para]=dplRNNTrain(nin, nh, nout, input, output, reg, lambda)
%
% INPUT nin: number of input unints
%		nh: number of hidden units
%		nout: number of output units
%		para: parameter vector
%		input: nin x T
%		ouput: nout x T
%		reg: L1 / L2
%		lambda: regularization parameter  
%
% OUTPUT cost:
%			grad:  
%
% REMARKS
%
% created with MATLAB ver.: 8.0.0.783 (R2012b) on Microsoft Windows 7 Version 6.1 (Build 7601: Service Pack 1)
%
% created by: Honglei Liu
% DATE: 23-Oct-2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options.Method = 'lbfgs'; 
options.maxIter =200;  
options.maxFunEvals = 100000;
options.display = 'on';
options.DerivativeCheck = 'off';

para=rand(nin*nh + nh*nh + nh*nout,1)-0.5;
[optpara, cost] = minFunc( @(p)  dplRNNGrad(nin,nh,nout,p,input,output,reg,lambda), para, options);

end
