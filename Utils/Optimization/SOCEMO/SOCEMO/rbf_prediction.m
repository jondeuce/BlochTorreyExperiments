function yhat = rbf_prediction(CandPoint,Data,lambda, gamma, rbf_flag )
%% rbf_prediction.m uses an RBF surrogate model to predict the function 
%values at the points in the set CandPoint
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%CandPoint - matrix with points at which we want to predict the function value
%Data - structure with all problem information
%lambda, gamma - parameters of RBF model
%rbf_flag - indicator of the type of RBF model that we want to fit (cub, lin, tps)
%--------------------------------------------------------------------------
%output:
%yhat - predicted objective function value 
%--------------------------------------------------------------------------

[mX,~]=size(CandPoint); %dimensions of the points where function value should be predicted
R = pdist2(CandPoint,Data.S); %compute pairwise dstances between points in CandPoint and S. pdist2 is MATLAB built-in function

if strcmp(rbf_flag,'cub') %cubic RBF
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear RBF
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end
   
p1 = Phi*lambda; %first part of response surface - weighted sum of distances
p2 = [CandPoint,ones(mX,1)]*gamma; % polynomial tail of response surface
yhat=p1+p2; %predicted function value

end%function
