function yhat = rbf_prediction_mo(CandPoint,Data,lambda, gamma, rbf_flag )
%%rbf_prediction_mo.m uses an RBF surrogate model to predict the objective 
%function values of CandPoint
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs:
%CandPoint - the point at which the objective function values shall be predicted
%Data - structure with all problem information
%lambda, gamma - matrices with RBF parameters for each objective (column number = objective number)
%rbf_flag - type of RBF surrogate we want (should be consistent throughout the algorithm)
%--------------------------------------------------------------------------
%outputs: 
%yhat - predicted objective function values
%--------------------------------------------------------------------------

[mX,~]=size(CandPoint); %dimensions of the points where function values should be predicted
R = pdist2(CandPoint,Data.S); %compute pairwise dstances between points in CandPoint and S. pdist2 is MATLAB built-in function

if strcmp(rbf_flag,'cub') %cubic RBF
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear RBF
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end

%multiobjective prediction 
yhat = zeros(mX,Data.nr_obj);
for ii = 1:Data.nr_obj
    p1 = Phi*lambda(:,ii); %first part of response surface - weighted sum of distances
    p2 = [CandPoint,ones(mX,1)]*gamma(:,ii); % polynomial tail of response surface
    yhat(:,ii)=p1+p2; %predicted function value
end

end%function
