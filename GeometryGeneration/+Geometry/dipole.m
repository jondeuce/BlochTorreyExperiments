function D = dipole(mSize, vSize, BDir, prec, kspace)
%DIPOLE Generate the Dipole Kernel in kspace
%
%   mSize - matrix size
%   vSize - voxel size in mm
%   BDir  - unit vector of B0 field
%
%   D - dipole kernel in kspace
    
    if nargin < 5
        kspace = true;
    end
    
%     if ~kspace
%         warning('Image space not implemented; defaulting to kspace');
%         kspace = true;
%     end

    if nargin == 4 && strcmpi(prec, 'single')
        mSize = single(mSize);
    end
    
    if isscalar(vSize)
        vSize = [vSize, vSize, vSize];
    end
    
    % Force column vector
    mSize = mSize(:);
    vSize = vSize(:);
    
    % Making sure both even and odd sizes will be centered properly
    mLow  = floor(mSize / 2);
    mHigh = mLow - ~mod(mSize, 2);
    
    % Step Size
    if kspace
        dxyz  = 1 ./ (mSize .* vSize);
    else
        dxyz  = vSize ./ mSize;
    end
    
    mLow  = mLow  .* dxyz;
    mHigh = mHigh .* dxyz;
    
    [X, Y, Z] = ndgrid(-mLow(1) : dxyz(1) : mHigh(1),...
                       -mLow(2) : dxyz(2) : mHigh(2),...
                       -mLow(3) : dxyz(3) : mHigh(3));
    
    if kspace
        kz = X.*BDir(1) + Y.*BDir(2) + Z.*BDir(3);
        k2 = X.*X + Y.*Y + Z.*Z;
        I = find(k2 == 0); % origin will be set to zero
        clear X Y Z
        
        D  = 1/3 - (kz.*kz ./ k2);
        clear kz k2
        
        if ~isempty(I)
            D(I) = 0;
        end
        
        D(abs(D) <= eps(2/3)) = 0;
        
        % c = floor(mSize ./ 2 + 1);
        % D(c(1), c(2), c(3)) = -2/3;
        
        D  = ifftshift(D);
    else
        rz = X.*BDir(1) + Y.*BDir(2) + Z.*BDir(3);
        r2 = X.*X + Y.*Y + Z.*Z;
        I = find(r2 == 0); % origin will be set to zero
        clear X Y Z
        
        num = 3 .* rz .* rz - r2;
        clear rz
        
        den = (4*pi) .* (r2 .* r2 .* sqrt(r2));
        clear r2
        
        D  = num ./ den;
        clear num den
        
        if ~isempty(I)
            D(I) = 0; % set origin to zero
        end
        
        D  = fftn(ifftshift(D));
    end
    
end
