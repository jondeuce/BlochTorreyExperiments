function y = BlochTorreyAction(x, h, D, f, gsize, iters, type)
%BLOCHTORREYACTION Discrete Bloch-Torrey operator w/ 2nd order
% central difference approximations on derivatives and periodic boundary
% conditions. The operator is:
%
%   L[x] = div(D*grad(x)) - Gamma*x
%        = D*lap(x) - Gamma*x          % for scalar D
%
% Currently, D must be a scalar and f must be a 3D array of the same size
% as x.
%
% INPUTS:
%   x:     input array (3D (or flattened 1D) or 4D (or flattened 2D) complex double array)
%   h:     grid spacing (scalar double)
%   D:     diffusion constant (scalar double)
%   f:     diagonal term = -6*D/h^2 - Gamma (3D complex double array)
%   gsize: size of grid operated on (3-element array)
%   iters: Number of iterations to apply the operator to the input (default 1)
%   type:  regular operator (default '') or conjugate transpose ('transp')

    if nargin < 7 || ~isa(type,'char'); type = ''; end

    if nargin < 6 || isempty(iters); iters = 1; end
    if ~( iters > 0 && iters == round(iters) ); error('iters must be a positive integer'); end
    
    gsize  = gsize(:).'; if numel(gsize) ~= 3; error('grid must be 3-dimensional'); end
    nslice = prod(gsize); %number of elems. on the 3D grid
    ntotal = numel(x); %total number of elements in x
    ntime  = round(ntotal/nslice); %number of 'time' slices (i.e. reps of the 3D array)

    if nslice * ntime ~= ntotal
        error('Incorrect dimensions: numel(x)/prod(gsize) must be an integer');
    end

    xsize   = size(x);
    gsize4D = [gsize, ntime]; %3D array repeated along 4th dimension
    gsize2D = [nslice, ntime]; %3D array flattened to column and repeated along 2nd dimension
    gsize1D = [nslice, 1]; %3d-grid flattened to column
    ndim    = 3 + (ntime>1); %Indicates whether we have 3D (flattened 1D) or 4D (flattened 2D) data

    if ~checkDims(x, ndim, gsize, gsize1D, gsize2D, gsize4D)
        % Allowable sizes are the repeated grid size in 4-D, or the flattened size in 2-D
        error('size(x) must be either the (repeated-)grid size or (repeated-)flattened size');
    end

    f = checkDiag(f, ndim, ntime, xsize, gsize, gsize1D, gsize2D, gsize4D);
    D = checkDiffusionCoeff(D);    
    h = checkGridSpacing(h);    

    %----------------------------------------------------------------------

    isSingle	=   isa(x,'single');
    isDouble	=   isa(x,'double');
    isReal      =   isreal(x);

    if ~(isSingle || isDouble), error('x must be double or single.'); end
    if isSingle, x = double(x); end
    if isReal, x = complex(x); end

    if strcmpi(type,'transp')
        y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);
    else
        y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);
    end

    if isSingle, y = single(y); end

end

function bool = checkDims(x, ndim, gsize, gsize1D, gsize2D, gsize4D)
    bool = ( ndim == 3 && checkDims3D(x, gsize, gsize1D) ) || ...
           ( ndim == 4 && checkDims4D(x, gsize2D, gsize4D) );
end

function bool = checkDims3D(x, gsize, gsize1D)
    bool = ( isequal(size(x), gsize) || isequal(size(x), gsize1D) );
end

function bool = checkDims4D(x, gsize2D, gsize4D)
    bool = ( isequal(size(x), gsize2D) || isequal(size(x), gsize4D) );
end

function f = checkDiag(f, ndim, ntime, xsize, gsize, gsize1D, gsize2D, gsize4D)

    if isempty(f)
        f = zeros(xsize,'double');
    elseif isscalar(f)
        f = double(f) * ones(xsize,'double');
    elseif checkDims3D(f, gsize, gsize1D)
        if ndim == 4
            if isequal(f,gsize1D)
                f = repmat(f,[1,1,1,ntime]);
            else
                f = repmat(f,[1,ntime]);
            end
        end
    elseif ~checkDims4D(f, gsize2D, gsize4D)
        error('size(f) must be one of: scalar, (repeated-)grid size, (repeated-)flattened size');
    end
    if ~isa(f,'double'), f = double(f); end
    if isreal(f); f = complex(f); end
    
end

function D = checkDiffusionCoeff(D)

    if isempty(D)
        D = 0;
    elseif ~(isscalar(D) && isfloat(D))
        error('D must be a floating point scalar');
    end
    %if ~(D >= 0); warning('D should (typically) be a positive number!'); end
    
    if ~isa(D,'double')
        D = double(D);
    end
    
end

function h = checkGridSpacing(h)

    if isempty(h)
        h = 1;
    elseif ~(isfloat(h) && all(h > 0))
        error('h must be a positive floating point scalar or 3-element array');
    end
    if ~isscalar(h)
        if ~( numel(h) == 3 && norm(diff(h)) < 5*max(eps(h(:))) )
            error('Subvoxels must 3D and be isotropic.');
        end
        h = h(1);
    end
    if ~isa(h,'double'); h = double(h); end

end