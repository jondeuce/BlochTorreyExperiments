function [DiOmega, Geom] = CalculateAnisotropicOmega(mSize,vSize,alpha,gyro,B0,CylinderRad,IsoChi,AniChi)
% Calculates the off-resonance frequency for different orientation and
% cylinder radious

    if nargin == 0
        [DiOmega, Geom] = runMinExample;
        return
    end

    [DiOmega, Geom] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,CylinderRad,IsoChi,AniChi);
    
end

function [DiOmega, Geom] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,CylinderRad,IsoChi,AniChi)
    %mSize: grid,  vSize: unitsTensorMap = zeros(ChiMapSize);
    
    BDir  = [sin(alpha); 0; cos(alpha)];
    
    % Force column vector
    mSize = mSize(:);
    vSize = vSize(:);
    
    % Making sure both even and odd sizes will be centered properly
    mLow  = floor(mSize / 2);
    mHigh = mLow - ~mod(mSize, 2);
    
    % Step Size
    dxyz  = 1 ./ (mSize .* vSize);
    
    mLow  = mLow  .* dxyz;
    mHigh = mHigh .* dxyz;
    
    % k =
    [KX, KY, KZ] = ndgrid(-mLow(1) : dxyz(1) : mHigh(1),...
                          -mLow(2) : dxyz(2) : mHigh(2),...
                          -mLow(3) : dxyz(3) : mHigh(3)); 
    
   
    KX = ifftshift(KX);
    KY = ifftshift(KY);
    KZ = ifftshift(KZ);
    k =  permute(cat(4, KX, KY, KZ),[4,1,2,3]);
    %kTrans = permute(k,[2,3,4,1]);
    
    k2 = KX.*KX + KY.*KY + KZ.*KZ;
    clear KX KY KZ
        
    [ChiMap, Geom] = Create_Geom_with_Tensor(mSize,vSize,CylinderRad,IsoChi,AniChi);
    
    FFTchi = fftnTensor(ChiMap); % FFT of tensor map 3x3xNxNxN
  
    %DiOmega = gyro * B0 * real(ifftn((1/3 * BDir' * fftnTensor(ChiMap) * BDir) - ((kz * k' * fftnTensor(ChiMap) * BDir / k2)));
    
    % first part
    
    % B'*ChiMap
    a1 = zeros([3,mSize.']);
    for ii = 1:mSize(1)
        for jj = 1:mSize(2)
            for kk = 1:mSize(3)
                a1(:,ii,jj,kk) = BDir' * FFTchi(:,:,ii,jj,kk);
            end
        end
    end
    
   
    % 1/3 *B'*ChiMap*B
    %a1perm = permute(a1,[3,4,5,1,2]);
    Apart = zeros(mSize.');
    for ii = 1:mSize(1)
        for jj = 1:mSize(2)
            for kk = 1:mSize(3)
                Apart(ii,jj,kk) = 1/3 * (a1(:,ii,jj,kk)' * BDir);
            end
        end
    end
    
    % second part
    
    % kz = KX.*BDir(1) + KY.*BDir(2) + KZ.*BDir(3);
    kz = zeros(mSize.');
    for ii = 1:mSize(1)
        for jj = 1:mSize(2)
            for kk = 1:mSize(3)
                kz(ii,jj,kk) = BDir' * k(:,ii,jj,kk);
            end
        end
    end
    
    % k'* ChiMap
    a3 = zeros([3,mSize.']);
    for ii = 1:mSize(1)
        for jj = 1:mSize(2)
            for kk = 1:mSize(3)
                %a3(:,ii,jj,kk) = squeeze(kTrans(ii,jj,kk,:))' * FFTchi(:,:,ii,jj,kk);
                a3(:,ii,jj,kk) = k(:,ii,jj,kk)' * FFTchi(:,:,ii,jj,kk);
            end
        end
    end
    
    % k'* FFTchi*B 
    %a3perm = permute(a3,[2,3,4,1]);
    a4 = zeros(mSize.');
    for ii = 1:mSize(1)
        for jj = 1:mSize(2)
            for kk = 1:mSize(3)
                %a4(ii,jj,kk) = squeeze(a3perm(ii,jj,kk,:))' * BDir;
                a4(ii,jj,kk) = a3(:,ii,jj,kk)' * BDir;
            end
        end
    end
    
    % B'*k * k'*FFTchi*B / k2
    tmp = Apart - (kz .* a4 ./ k2);
    tmp(k2 == 0) = 0;
  
    % 1/3 *B'*ChiMap*B - B'*k k'*FFTchi*B / k2
    DiOmega = gyro * B0 * real(ifftn(tmp));

end

function [ChiMap, Geom] = Create_Geom_with_Tensor(mSize,vSize,CylinderRad,IsoChi,AniChi)

[X, Y, Z] = ndgrid(linspacePeriodic(-vSize(1)/2,vSize(1)/2, mSize(1)),...
                   linspacePeriodic(-vSize(2)/2,vSize(2)/2, mSize(2)),...
                   linspacePeriodic(-vSize(3)/2,vSize(3)/2, mSize(3)));
                      
%boolean cylinder
% Geom = (randn(mSize(:).')>0);
Geom = (X.^2 + Y.^2 < CylinderRad^2 & abs(Z) < vSize(3)/4 );

ChiMap = ApplyTensor(Geom,X,Y,IsoChi,AniChi);

end


function [TensorMap] = ApplyTensor(Geom,X,Y,IsoChi,AniChi)
% Map is NxNxN

MapSize = size(Geom);
ChiMapSize = [3,3,MapSize];

TensorMap = zeros(ChiMapSize);
[sz1,sz2,sz3] = size(Geom);
for ii = 1:sz1
    for jj = 1:sz2
        for kk = 1:sz3
            fi = atan2(Y(ii,jj,kk),X(ii,jj,kk));
            % apply susceptibility only on cylinder points
            if Geom(ii,jj,kk)
                TensorMap(:,:,ii,jj,kk) = Tensor(fi,IsoChi,AniChi);
            else 
                TensorMap(:,:,ii,jj,kk) = 0;
            end
        end
    end    
end

end

function ChiT = Tensor(fi,IsoChi,AniChi)

ChiT = [cos(fi), sin(fi), 0;
       -sin(fi), cos(fi), 0;
              0,       0, 1] * ...
       [IsoChi+AniChi,                0,                0;
                    0, IsoChi-AniChi./2,                0;
                    0,                0, IsoChi-AniChi./2] * ...
       [cos(fi), -sin(fi), 0;
        sin(fi),  cos(fi), 0;
              0,        0, 1]; 

end

function [Y] = fftnTensor(TensorMap)
% X is 3x3xNxNxN

Y = TensorMap;
for ii = 1:3
    for jj = 1:3
      Y(ii,jj,:,:,:) = fftn(TensorMap(ii,jj,:,:,:));
    end
end

end



function [DiOmega,Geom] = runMinExample
% Example which shows a cylinder...

mSize = [100,100,100];
vSize = [1,1,1];
alpha = pi/2;
gyro = 2.67515255e8;  % Gyromagnetic ratio [rad/(T*s)]
B0 = -3.0;
CylinderRad = 0.35;
IsoChi = -60e-9;   % isotropic magnetic susceptibility
%AniChi = -120e-9;  % anisotropic magnetic susceptibility
AniChi = 0;  % anisotropic magnetic susceptibility

[DiOmega, Geom] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,CylinderRad,IsoChi,AniChi);
% imagesc(DiOmega(:,:,1));
% global VoxelGeom;
% VoxelGeom = Geom;

end
