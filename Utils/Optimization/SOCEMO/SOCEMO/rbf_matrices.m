function [PHI,P,phi0, pdim]=  rbf_matrices(Data,rbf_flag)
%rbf_matrices.m set up the matrices needed for solving the linear system in
%order to get the parameters of the RBF surrogate
%--------------------------------------------------------------------------
%Author information
%Juliane Mueller
%juliane.mueller2901@gmail.com
%--------------------------------------------------------------------------
%inputs:
%Data - structure with all problem information
%rbf_flag - the type of RBF surrogate we want to fit (cubic, linear,
%thin-plate spline); must be consistent
%--------------------------------------------------------------------------
%output:
%PHI - matrix with RBF values of distances
%P - sample site matrix with column vector of 1 at the end (for polynomial
%tail)
%phi0 - RBF value for distance 0
%pdim - number of parameters to determine for polynomial tail 
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

PHI(1:Data.m,1:Data.m)=PairwiseDistance; %matrix containing RBF values of pairwise distances
phi0 = rbfvalue(0, rbf_flag); %phi-value where distance of 2 points =0 (diagonal entries)
pdim = Data.dim + 1;
P = [Data.S,ones(Data.m,1)];% [S,ones(maxevals,1)];

end %function


