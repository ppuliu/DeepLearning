
nin=70;
nh=5;
nout=70;
lambda=1.6;
T=100;
X=rand(nin,T);

input=X(:,1:T-1);
output=X(:,2:T);

para=rand(nin*nh + nh*nh + nh*nout,1)-0.5;

tic
numgrad=computeNumericalGradient( @(p) dplRNNGrad(nin,nh,nout,p,input,output,'L1',lambda), para);
[~, grad]=dplRNNGrad(nin,nh,nout,para,input,output,'L1',lambda);
disp([numgrad grad]); 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff);  % Should be small. In our implementation, these values are
             % usually less than 1e-9.
             % When you got this working, Congratulations!!! 

disp('Verification Complete!');
toc

%%
para=rand()-0.5;

tic
numgrad=computeNumericalGradient( @(p) dplActivationFunc(p, 'tanh'), para);
[~, grad]=dplActivationFunc(para, 'tanh');
disp([numgrad grad]); 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff);  % Should be small. In our implementation, these values are
             % usually less than 1e-9.
             % When you got this working, Congratulations!!! 

disp('Verification Complete!');
toc