function y = BlochTorreyAction(x, h, D, f, gsize3D, iters, istrans, isdiag, mask)
%BLOCHTORREYACTION Discrete Bloch-Torrey operator w/ 2nd order
% central difference approximations on derivatives and periodic boundary
% conditions. The operator is:
%
%   L[x] = div(D*grad(x)) - Gamma*x
%        = D*lap(x) - Gamma*x          % for scalar D
%
% D and f must be either scalars or 3D arrays of size gsize3D.
%
% INPUTS:
%   x:          Input array (3D (or flattened 1D) or 4D (or flattened 2D) complex double array)
%   h:          Grid spacing (scalar double)
%   D:          Diffusion constant (scalar double or 3D array)
%   f:          Represents either the complex decay Gamma = R2+i*dw, or the
%               isotropic diagonal term f = -6*D/h^2 - Gamma (3D complex double array)
%   gsize3D:    Size of grid operated on (3-element array)
%   iters:      Number of iterations to apply the operator to the input (default 1)
%   istrans:    Applies conjugate transpose operator if true (default false)
%   isdiag:     Assumes f represents the diagonal term (default true)

DEBUG = false;
warn = @(str) warn_(DEBUG, str);

if nargin < 9; mask = logical([]); end
if ~(isa(mask, 'logical') || isa(mask, 'double'))
    warning('converting mask to double');
    mask = double(mask);
end

if nargin < 8 || ~isa(isdiag, 'logical'); isdiag  = true; end
if nargin < 7 || ~isa(istrans,'logical'); istrans = false; end

if nargin < 6 || isempty(iters); iters = 1; end
if ~(iters > 0 && iters == round(iters)); error('iters must be a positive integer'); end

[ndim, ntime, gsize1D, gsize2D, gsize3D, gsize4D] = getGridSizes(x, gsize3D);

x = checkArray(x, gsize1D, gsize2D, gsize3D, gsize4D, true,  true);
f = checkArray(f, gsize1D, gsize2D, gsize3D, gsize4D, true,  true);
D = checkArray(D, gsize1D, gsize2D, gsize3D, gsize4D, false, true);
h = checkGridSpacing(h);

%----------------------------------------------------------------------

isSingle = isa(x,'single');
isDouble = isa(x,'double');
isReal   = isreal(x);

if ~(isSingle || isDouble), error('x must be double or single.'); end
if isSingle, x = double(x); warn('BLOCHTORREYACTION: converting to double'); end
if isReal, x = complex(x); warn('BLOCHTORREYACTION: converting to complex'); end

if istrans
    if ~isreal(f); f = conj(f); warn('BLOCHTORREYACTION: conjugating'); end
    if ~isreal(D); D = conj(D); warn('BLOCHTORREYACTION: conjugating'); end
end

if isscalar(D) && ~isempty(mask)
    D = D * ones(gsize3D); warn('BLOCHTORREYACTION: expanding scalar D to gsize');
end

if isscalar(D)
    if isscalar(f)
        % Avoid allocating f*ones by just taking laplacian
        Gamma = f;
        if isdiag; Gamma = -6*D/h^2 - Gamma; end
        y = D*Laplacian(x, h, gsize3D, iters) - Gamma*x;
    else
        if ~isreal(D)
            % BlochTorreyAction_cd assumes real D
            y = D*BlochTorreyAction_cd(x, h, 1, f/D, gsize4D, ndim, iters, isdiag); warn('BLOCHTORREYACTION: complex scalar D');
        else
            y = BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters, isdiag);
        end
    end
else
    if isscalar(f)
        f = f * ones(gsize3D); warn('BLOCHTORREYACTION: expanding gamma to gsize');
    end
    if isempty(mask)
        if ~isreal(D)
            %TODO this is inefficient allocation-wise
            warn('BLOCHTORREYACTION: complex D array');
            y = BTActionVariableDiff_cd(x, h, real(D), f, gsize4D, ndim, iters, isdiag);
            y = y + 1i * BTActionVariableDiff_cd(x, h, imag(D), complex(zeros(size(f))), gsize4D, ndim, iters, isdiag);
        else
            y = BTActionVariableDiff_cd(x, h, D, f, gsize4D, ndim, iters, isdiag);
        end
    else
        if islogical(mask)
            BTNeumann = @BTActionVariableDiffNeumannBoolMask_cd;
        else
            BTNeumann = @BTActionVariableDiffNeumann_cd;
        end
        if ~isreal(D)
            %TODO this is inefficient allocation-wise
            warn('BLOCHTORREYACTION: complex D array');
            y = BTNeumann(x, h, real(D), f, gsize4D, ndim, iters, isdiag, mask);
            y = y + 1i * BTNeumann(x, h, imag(D), complex(zeros(size(f))), gsize4D, ndim, iters, isdiag, mask);
        else
            y = BTNeumann(x, h, D, f, gsize4D, ndim, iters, isdiag, mask);
        end
    end
end

if isSingle, y = single(y); warn('BLOCHTORREYACTION: converting output to single');end

end

function warn_(DEBUG, str)
if DEBUG
    warning(str);
end
end

% ----------------------------------------------------------------------- %
% Old transpose code
% ----------------------------------------------------------------------- %

% if isscalar(D)
%     if isscalar(f)
%         % Avoid allocating f*ones by just taking laplacian
%         Gamma = f;
%         if isdiag; Gamma = conj(-6*D/h^2 - Gamma); end
%         y = D*Laplacian(x, h, gsize4D, iters) - Gamma*x;
%     else
%         if ~isdiag
%             f = conj(-6*D/h^2 - f); % f represents Gamma
%         end
%         if ~isreal(D)
%             % trans_BlochTorreyAction_cd assumes real D
%             y = D*trans_BlochTorreyAction_cd(x, h, 1, f/D, gsize4D, ndim, iters);
%         else
%             y = trans_BlochTorreyAction_cd(x, h, D, f, gsize4D, ndim, iters);
%         end
%     end
% else
%     if isscalar(f)
%         warning('TODO: expanding Gamma to size of grid');
%         f = f*ones(gsize3D);
%     end
%     if ~isreal(D)
%         %TODO this is somewhat inefficient allocation-wise
%         y = BTActionVariableDiff_cd(x, h, real(D), conj(f), gsize4D, ndim, iters, isdiag);
%         y = y + 1i * BTActionVariableDiff_cd(x, h, -imag(D), complex(zeros(size(f))), gsize4D, ndim, iters, isdiag);
%     else
%         y = BTActionVariableDiff_cd(x, h, D, conj(f), gsize4D, ndim, iters, isdiag);
%     end
% end

% ----------------------------------------------------------------------- %
% Old array size checking code
% ----------------------------------------------------------------------- %

% function bool = checkDims(x, ndim, gsize1D, gsize2D, gsize3D, gsize4D)
% bool = ( ndim == 3 && checkDims3D(x, gsize1D, gsize3D) ) || ...
%     ( ndim == 4 && checkDims4D(x, gsize2D, gsize4D) );
% end
%
% function bool = checkDims3D(x, gsize1D, gsize3D)
% bool = ( isequal(size(x), gsize3D) || isequal(size(x), gsize1D) );
% end
%
% function bool = checkDims4D(x, gsize2D, gsize4D)
% bool = ( isequal(size(x), gsize2D) || isequal(size(x), gsize4D) );
% end
%
% function f = checkArray(f, gsize1D, gsize2D, gsize3D, gsize4D, forcecplx)
%
% if isempty(f)
%     f = 0;
% elseif isscalar(f)
%     f = double(f);
% elseif ~( checkDims3D(f, gsize1D, gsize3D) || checkDims4D(f, gsize2D, gsize4D) )
%     error('size(f) must be one of: scalar, (repeated-)grid size, (repeated-)flattened size');
% end
%
% if ~isa(f,'double'), f = double(f); end
% if isreal(f) && forcecplx; f = complex(f); end
%
% end
%
% function h = checkGridSpacing(h)
%
% if isempty(h)
%     h = 1;
% elseif ~(isfloat(h) && all(h > 0))
%     error('h must be a positive floating point scalar or 3-element array');
% end
% if ~BlochTorreyOp.is_isotropic(h)
%     error('h must be isotropic');
% end
%
% h = h(1);
% if ~isa(h,'double'); h = double(h); end
%
% end