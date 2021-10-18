function [lambda, gamma] = rbf_params_mo(Data, rbf_flag)
%rbf_params_mo.m computes the RBF parameters for each objective function
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%input:
%Data - structure with all up-to-date information
%rbf_flag - string determining what kind of RBF we want to fit to the data
%--------------------------------------------------------------------------
%output:
%lambda, gamma - RBF parameters, column number corresponds to objective function number
%--------------------------------------------------------------------------

distances=pdist2(Data.S,Data.S); %compute pairwise dstances between points in S, pdist2 is MATLAB built-in function
if strcmp(rbf_flag,'cub') %cubic RBF
    PairwiseDistance=distances.^3; 
elseif strcmp(rbf_flag,'lin') %linear RBF
    PairwiseDistance=distances;
elseif strcmp(rbf_flag,'tps') %thin-plate spline RBF
    PairwiseDistance=distances.^2.*log(distances+realmin);
    PairwiseDistance(logical(eye(size(distances)))) = 0;
end

PHI(1:Data.m,1:Data.m)=PairwiseDistance; %matrix with RBF values of distances
P = [Data.S,ones(Data.m,1)];% matrix with observation sites and appended column of 1s
[m,n]=size(Data.S); %determine how many points are in S and what is the dimension
A=[PHI,P;P', zeros(n+1,n+1)]; %set up matrix for solving for parameters
lambda = zeros(m,Data.nr_obj); %initialize matrices for parameters
gamma = zeros(Data.dim+1, Data.nr_obj);

for ii = 1:Data.nr_obj %compute paramters for each objective iteratively
   RHS=[Data.Y(:,ii);zeros(n+1,1)]; %right hand side of linear system
   params=A\RHS; %compute parameters
   lambda(:,ii)=params(1:m); %parameters in weighted sum part of RBF
   gamma(:,ii)=params(m+1:end); %parameters of polynomial tail	
end

end %function
