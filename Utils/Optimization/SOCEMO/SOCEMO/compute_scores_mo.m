function xnew = compute_scores_mo(Data,CandPoint, w_r, lambda, gamma, rbf_flag )
%compute_scores_mo.m computes the score for perturbation points based on 
%surrogate model predictions and the distance of the perturbation points to
%already evaluated points
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs: 
%Data - Data structure that contains information of the problem 
%CandPoint - Candidate point created by pertubation
%w_r - weight we put on predicted objective function values; (1-w_r) = weight we put on distance criterion
%lambda, gamma - RBF parameters of objective functions
%rbf_flag - indicates what type of RBF we want (cub, lin, tps)
%--------------------------------------------------------------------------
%outputs:
%xnew - new sample site
%--------------------------------------------------------------------------
  
[mX,~]=size(CandPoint); %dimensions of the points where function value should be predicted
R = pdist2(CandPoint,Data.S); %compute pairwise dstances between points in X and S. pdist2 is MATLAB built-in function

%check which kind of RBF
if strcmp(rbf_flag,'cub') %cubic
    Phi=R.^3;
elseif strcmp(rbf_flag,'lin') %linear
    Phi=R;
elseif strcmp(rbf_flag,'tps') %thin-plate spline
    Phi=R.^2.*log(R+realmin);
    Phi(logical(eye(size(R)))) = 0;
end

%predict objective function values for all candidates. For each objective function, scale the function values to [0,1]
%(for easier comparison of values of different ranges)
predYscaled = zeros(mX,Data.nr_obj);
for ii = 1:Data.nr_obj
    p1 = Phi*lambda(:,ii); %first part of response surface - weighted sum of distances
    p2 = [CandPoint,ones(mX,1)]*gamma(:,ii); % polynomial tail of response surface
    yhat=p1+p2; %predicted function value

    %scale predicted objective function values to [0,1]
    min_yhat = min(yhat); %find min of predIcted objective function value
    max_yhat = max(yhat); %find maximum of predicted objective function value
    if min_yhat == max_yhat  %compute scaled objective function value scores
        scaled_yhat=ones(length(yhat),1);
    else
        scaled_yhat = (yhat-min_yhat)/(max_yhat-min_yhat);
    end
    predYscaled(:,ii) = scaled_yhat;
end
dist=pdist2(CandPoint,Data.S(1:Data.m,:))';

%valueweight=w_r; %weight for response surface criterion 
%scale distances to already evaluated points to [0,1]
min_dist = (min(dist,[],1))'; %minimum distance of every candidate point to already sampled points  
max_min_dist = max(min_dist); %maximum of distances
min_min_dist = min(min_dist); %minimum of distances
if  max_min_dist == min_min_dist  %compute distance criterion scores
     scaled_dist =ones(length(min_dist),1);
else
     scaled_dist = (max_min_dist-min_dist)/(max_min_dist-min_min_dist);
end

%compute weighted score for all candidates
%assume all objectives have the same influence, use average of sclaed function values
score = w_r*mean(predYscaled,2) + (1 -w_r)*scaled_dist;

%assign bad scores to candidate points that are too close to already sampled
%points
score(min_dist < Data.tol_same) = Inf; 
%find candidate with best score -> becomes new sample point
[~,minID] = min(score);
xnew = CandPoint(minID,:);  %new sample point


end %function
