function [paths] = genPath(p,varargin)
%GENPATH Wrapper for genpath_exclude for the 1 argument case
    
    if nargin <= 1
        paths = genpath(p);
    else
        paths = genpath_exclude(p,varargin{:});
    end
    
end

