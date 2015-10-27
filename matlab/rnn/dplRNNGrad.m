function [cost,grad]=dplRNNGrad(nin,nh,nout,para,input,output,reg,lambda)
%DPLRNNGRAD calculates the cost and gradients of parameters for one scan
%
% SYNOPSIS: [cost, grad]=dplRNNGrad(nin,nh,nout,para,input,output,lamda=1)
%
% INPUT nin : number of input units
%		nh : number of hidden units
%		nout : number of output units
%		para : current values of parameters (nh*nin+nh^2+nout*nh) x 1
%		input : input tensor 	nin x T
%		output : output tensor	nout x T
%		lamda : regularization term                                     
%
% OUTPUT grad : gradient of para	(nin*nh+nh**2+nh*nout) x 1
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

lossFuncName='binary_crossentropy';

if ~(ndims(input)==ndims(output))
    error('input and output dimentions must agree');
end

[tnin, T1]=size(input);

if ~(tnin==nin)
    error('input units must agree with nin');
end

[tnout,T2]=size(output);

if ~(tnout==nout)
    error('output units must agree with nout');
end

if ~(T1==T2)
    error('input and output must have the same T');
end

T=T1;

if ~(length(para)==nin*nh+nh*nh+nh*nout)
    error('para dimention does not match');
end

if ~exist('lambda','var')
    lambda=1;
end

if ~exist('reg','var')
    reg='L1';
end
    

idx=1;
wIn=reshape(para(idx:idx+nin*nh-1),[nh,nin]); % nh x nin
wInGrad=zeros(size(wIn));
idx=idx+nin*nh;
wH=reshape(para(idx:idx+nh*nh-1),[nh,nh]);  % nh x nh
wHGrad=zeros(size(wH));
idx=idx+nh*nh;
wOut=reshape(para(idx:idx+nh*nout-1),[nout,nh]); % nout x nh
wOutGrad=zeros(size(wOut));

% forward pass
cost=0;
diff=zeros(nout,T);
h=zeros(nh,T);
h0=zeros(nh,1); % initial hindden state

hGrad=zeros(nh,T);
dGrad=zeros(nout,T);

for t=1:T
    htm1=h0;
    if(t>1)
        htm1=h(:,t-1);
    end
    [ht,htGrad]=dplActivationFunc(wIn*input(:,t)+wH*htm1,'tanh');
    [dt,dtGrad]=dplActivationFunc(wOut*ht,'sigmoid');
    [lt,ltGrad]=dplLossFunc(dt,output(:,t),lossFuncName);    % loss function
    
    % cache the values
    cost=cost+lt;
    diff(:,t)=ltGrad;
    h(:,t)=ht;
    hGrad(:,t)=htGrad;
    dGrad(:,t)=dtGrad;
end


% calculate gradients
theta=zeros(nh,T);

t=T;
while(t>0)
    %disp(t);
    theta_tp1=zeros(nh,1);  % nh x 1
    if(t<T)
        theta_tp1=theta(:,t+1);
    end
    
    theta_current=(wOut'*(diff(:,t).*dGrad(:,t))).*hGrad(:,t); % nh x 1
    theta_t=theta_current+(wH'*theta_tp1).*hGrad(:,t); % attention, it's wH'
    
    theta(:,t)=theta_t;

    if(t>1)
        wHG_t=theta_t*h(:,t-1)';
    else
        wHG_t=theta_t*h0';
    end
    wInG_t=theta_t*input(:,t)';
    wOutG_t=(diff(:,t).*dGrad(:,t))*h(:,t)';
    
    % sum them with previous gradients
    wHGrad=wHGrad+wHG_t;
    wInGrad=wInGrad+wInG_t;
    wOutGrad=wOutGrad+wOutG_t;
    
    t=t-1;
end
grad=wInGrad(:);
grad=cat(1,grad,wHGrad(:));
grad=cat(1,grad,wOutGrad(:));

if strcmp(reg,'L1')
    disp('using L1 regularization');
    cost=cost+lambda*(sum(abs(para)));
    grad=grad+lambda*((para>0)*2-1);
else if strcmp(reg,'L2')
        disp('using L2 regularization');
        cost=cost+lambda*(para'*para)/2;
        grad=grad+lambda*para;
    end
end

    
end




    
    
