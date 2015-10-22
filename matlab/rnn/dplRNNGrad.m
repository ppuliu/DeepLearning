function [grad, cost]=dplRNNGrad(nin,nh,nout,para,input,output,lamda)
%DPLRNNGRAD calculates the cost and gradients of parameters for one scan
%
% SYNOPSIS: [grad, cost]=dplRNNGrad(nin,nh,nout,para,input,output,lamda=1)
%
% INPUT nin : number of input units
%		nh : number of hidden units
%		nout : number of output units
%		para : current values of parameters	1 x (nin*nh+nh**2+nh*nout)
%		input : input tensor 	time x nin x batch size
%		output : output tensor	time x nout x batch size
%		lamda : regularization term                                     
%
% OUTPUT grad : gradient of para	1 x (nin*nh+nh**2+nh*nout)
%			cost : objective value                              
%
% REMARKS
%
% created with MATLAB ver.: 8.0.0.783 (R2012b) on Microsoft Windows 7 Version 6.1 (Build 7601: Service Pack 1)
%
% created by: Honglei Liu
% DATE: 14-Oct-2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~(ndims(input)==ndims(output))
    error('input and output dimentions must agree');
end

[T1,tnin, nBatch1]=size(input);

if ~(tnin==nin)
    error('input units must agree with nin');
end

[T2,tnout,nBatch2]=size(output);

if ~(tnout==nout)
    error('output units must agree with nout');
end

if ~(T1==T2 && nBatch1==nBatch2)
    error('input and output must have the same T and batch size');
end

if ~(length(para)==nin*nh+nh*nh+nh*nout)
    error('para dimention does not match');
end

lamda=1;

idx=1;
wIn=reshape(para(idx:idx+nin*nh-1),[nin,nh]);
wInGrad=zeros(size(wIn));
idx=idx+nin*nh;
wH=reshape(para(idx:idx+nh*nh-1),[nh,nh]);
wHGrad=zeros(size(wH));
idx=idx+nh*nh;
wOut=reshape(para(idx:idx+nh*nout-1),[nh,nout]);
wOutGrad=zeros(size(wOut));
idx=idx+nh*nout-1;

h0=zeros(1,nh); % initial hindden state

grad=zeros(1,nin*nh+nh*nh+nh*nout);
%predicts=zeros(size(output));
cost=0;

for bidx=1:nBatch1
    bin=reshape(input(:,:,bidx),[T1,nin]);
    bh=zeros(T1,nh);   % T x nh
    bhGrad=zeros(T1,nh);    % cache the gradients for hidden states
    htm1=h0;
    for tidx=1:T1
        [ht,htGrad]=dplActivationFunc(bin(tidx,:)*wIn+htm1*wH,'tanh');
        bh(tidx,:)=ht;
        bhGrad(tidx,:)=htGrad;
        htm1=ht;
    end
    [bout,boutGrad]=dplActivationFunc(bh*wOut,'sigmoid'); % T x nout
    %predicts(:,:,bidx)=bout;
    
    % for this batch
    % calculate cost and gradents for parameters
    % use mean square error lost 
    diff=bout-output(:,:,bidx); % T x nout
    cost=cost+(diff(:)'*diff(:))/2;
    
    wOutGrad=wOutGrad+sum(diff(:))*bh'*boutGrad;
    wHGrad=wHGrad+sum(diff(:))*bh'*boutGrad*wOut';
    
end




    
    
