function y = BlochTorreyAction(x, h, D, f, gsize, iters, istrans, isdiag)
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
%   x:          Input array (3D (or flattened 1D) or 4D (or flattened 2D) complex double array)
%   h:          Grid spacing (scalar double)
%   D:          Diffusion constant (scalar double)
%   f:          Represents either the complex decay Gamma = R2+i*dw, or the
%               isotropic diagonal term f = -6*D/h^2 - Gamma (3D complex double array)
%   gsize:      Size of grid operated on (3-element array)
%   iters:      Number of iterations to apply the operator to the input (default 1)
%   istrans:    Applies conjugate transpose operator if true (default false)
%   isdiag:     Assumes f represents the diagonal term (default true)

if nargin < 8 || ~isa(isdiag, 'logical'); isdiag  = true; end
if nargin < 7 || ~isa(istrans,'logical'); istrans = false; end

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

f = checkArray(f, ndim, ntime, xsize, gsize, gsize1D, gsize2D, gsize4D, true);
D = checkArray(D, ndim, ntime, xsize, gsize, gsize1D, gsize2D, gsize4D, false);
h = checkGridSpacing(h);

%----------------------------------------------------------------------

isSingle	=   isa(x,'single');
isDouble	=   isa(x,'double');
isReal      =   isreal(x);

if ~(isSingle || isDouble), error('x must be double or single.'); end
if isSingle, x = double(x); end
if isReal, x = complex(x); end

if istrans
    if isscalar(D)
        if isscalar(f)
            % Avoid allocating f*ones by just taking laplacian
            Gamma = f;
            if isdiag; Gamma = conj(-6*D/h^2 - Gamma); end
            y = D*Laplacian(x, h, gsize4D, iters) - Gamma*x;
        else
            if ~isdiag
                f = conj(-6*D/h^2 - f); % f represents Gamma
            end
            if ~isreal(D)
                % trans_BlochTorreyAction_cd assumes real D
                y = D*trans_BlochTorreyAction_cd(x, h, 1, f/D, gsize4D, ndim, iters);
            else
                y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);
            end
        end
    else
        %TODO
        error('non-scalar D conjugate transpose not implemented');        
    end
else
    if isscalar(D)
        if isscalar(f)
            % Avoid allocating f*ones by just taking laplacian
            Gamma = f;
            if isdiag; Gamma = -6*D/h^2 - Gamma; end
            y = D*Laplacian(x, h, gsize, iters) - Gamma*x;
        else
            if ~isdiag
                f = -6*D/h^2 - f; % f represents Gamma
            end
            if ~isreal(D)
                % BlochTorreyAction_cd assumes real D
                y = D*BlochTorreyAction_cd(x, h, 1, f/D, gsize4D, ndim, iters);
            else
                y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);
            end
        end
    else
        if isdiag
            error('For non-scalar diffusion coefficient D, "f" must represent Gamma = R2 + i*dw');
        end
        if isscalar(f)
            warning('TODO: expanding Gamma to size of grid');
            f = f*ones(xsize);
        end
        if ~isreal(D)
            %TODO this is very inefficient
            Z = complex(zeros(size(f))); %need to use zeros for one input
            y = 1i * BTActionVariableDiff_cd(x, h, imag(D), Z, gsize4D, ndim, iters) + ...
                BTActionVariableDiff_cd(x, h, real(D), f, gsize4D, ndim, iters);
        else
            y = BTActionVariableDiff_cd(x, h, D, f, gsize4D, ndim, iters);
        end
    end
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

function f = checkArray(f, ndim, ntime, xsize, gsize, gsize1D, gsize2D, gsize4D, forcecplx)

if isempty(f)
    f = 0;
elseif isscalar(f)
    f = double(f);
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
if isreal(f) && forcecplx; f = complex(f); end

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