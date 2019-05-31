function D = dipole(mSize, vSize, BDir, prec, kspace)
%DIPOLE Generate the Dipole Kernel in kspace
% 
% INPUT
%   mSize  - matrix size
%   vSize  - voxel size in mm
%   BDir   - unit vector of B0 field
%   prec   - precision (defaults to 'double')
%   kspace - form dipole directly in kspace if true, or form in image
%            space and Fourier transform if false (default)
%
% OUTPUT
%   D      - dipole kernel in kspace

if nargin < 5; kspace = false; end
if nargin < 4; prec = 'double'; end

if isscalar(vSize)
    vSize = [vSize, vSize, vSize];
end

if strcmpi(prec, 'single')
    mSize = single(mSize);
    vSize = single(vSize);
    BDir = single(BDir);
end

% Force row vector
mSize = mSize(:).';
vSize = vSize(:).';

% Making sure both even and odd sizes will be centered properly
mLow = floor(mSize / 2);
mHigh = mLow - ~mod(mSize, 2);

% Step Size
dxyz = ones(3,1);
% if kspace
%   dxyz = 1 ./ (mSize .* vSize);
% else
%   dxyz = vSize ./ mSize;
% end

[X, Y, Z] = ndgrid( ...
    dxyz(1) .* (-mLow(1) : mHigh(1)), ...
    dxyz(2) .* (-mLow(2) : mHigh(2)), ...
    dxyz(3) .* (-mLow(3) : mHigh(3)));

% Index of center
Ic = sub2ind(mSize, mLow(1)+1, mLow(2)+1, mLow(3)+1);

if kspace
    kz = X.*BDir(1) + Y.*BDir(2) + Z.*BDir(3);
    k2 = X.*X + Y.*Y + Z.*Z;
    clear X Y Z
    
    D = 1/3 - (kz.*kz ./ k2);
    clear kz k2
    
    D(Ic) = 0; % set origin to zero
    % D(Ic) = -2/3; % set origin to -2/3
    % D(abs(D) <= eps(2/3)) = 0;
    
    D = ifftshift(D);
else
    rz = X.*BDir(1) + Y.*BDir(2) + Z.*BDir(3);
    r2 = X.*X + Y.*Y + Z.*Z;
    clear X Y Z
    
    num = 3 .* rz .* rz - r2;
    clear rz
    
    den = (4*pi) .* (r2 .* r2 .* sqrt(r2));
    clear r2
    
    D = num ./ den;
    clear num den
    
    D(Ic) = 0; % set image-space origin to zero (Lorentz sphere correction)
    
    D = fftn(ifftshift(D));
    
    D(1) = 0; % force k-space origin to zero
    % D(1) = -2/3; % force k-space origin to -2/3
end

end
