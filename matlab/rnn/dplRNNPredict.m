function [predictX, AUC]=dplRNNPredict(para, nin, nh, nout, input, plotfig, target)
%do predictions using trained rnn
%
% SYNOPSIS: [predictX, AUC]=dplRNNPredict(para, nin, nh, nout, input, start_step, plotfig)
%
% INPUT para: learned parameters
%		nin: number of input units
%		nh: number of hidden units
%		input: nin x T
%		plotfig: plot figure or not     
%		target: the target to predict nout x T                         
%
% OUTPUT predictX: nout x T
%
% REMARKS
%
% created with MATLAB ver.: 8.0.0.783 (R2012b) on Microsoft Windows 7 Version 6.1 (Build 7601: Service Pack 1)
%
% created by: Honglei Liu
% DATE: 23-Oct-2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~(length(para)==nin*nh + nh*nh + nh*nout)
    error('dimensions do not match')
end

if ~exist('plotfig','var')
    plotfig=false;
end

idx=1;
wIn=reshape(para(idx:idx+nin*nh-1),[nh,nin]); % nh x nin
idx=idx+nin*nh;
wH=reshape(para(idx:idx+nh*nh-1),[nh,nh]);  % nh x nh
idx=idx+nh*nh;
wOut=reshape(para(idx:idx+nh*nout-1),[nout,nh]); % nout x nh


% forward pass
[~,T]=size(input);
predictX=zeros(nout, T);
h=zeros(nh,T);
h0=zeros(nh,1); % initial hindden state

for t=1:T
    htm1=h0;
    if(t>1)
        htm1=h(:,t-1);
    end
    [ht,~]=dplActivationFunc(wIn*input(:,t)+wH*htm1,'tanh');
    [dt,~]=dplActivationFunc(wOut*ht,'sigmoid');
    
    % cache the values
    predictX(:,t)=dt;
    
end

AUC=0;
if exist('target','var')
    % calculate AUC
    labels=logical(target(:));
    scores=predictX(:);
    [AUC,fpr,tpr] = fastAUC(labels,scores,plotfig);
    
    if plotfig
        % plot overlay figure
        figure
        overlayPredicts(target,predictX);
    end
end

end