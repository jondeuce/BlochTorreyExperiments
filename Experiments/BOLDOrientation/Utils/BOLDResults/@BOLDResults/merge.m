function [ T ] = merge( R, S )
%MERGE Merge two BOLDResults arrays R and S. If R and S contain any common
%parameter sets, the results in R will be kept.

% Find common arguments
Targs = getcommonargs(R,S);
T = BOLDResults(Targs{:});

% Push S first, then R. Results in R will overwrite those in S, if any
% overlapping parameter sets exist.
T = push(T,S);
T = push(T,R);

end