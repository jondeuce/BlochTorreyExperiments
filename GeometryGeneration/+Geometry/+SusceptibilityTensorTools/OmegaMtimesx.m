function [DiOmega,Geom,X,Y] = OmegaMtimesx(mSize,vSize,alpha,gyro,B0,Rin,Rout,IsoChi,AniChi)
% Calculates the off-resonance frequency for different orientation and
% cylinder radious

    if nargin == 0
        
        [DiOmega,Geom,X,Y] = runMinExample;
        return
    end

    [DiOmega,Geom,X,Y] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,Rin,Rout,IsoChi,AniChi);
    
end

function [DiOmega,Geom,X,Y] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,Rin,Rout,IsoChi,AniChi)
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
    
   
%     % MATLAB fft shift
     KX = ifftshift(KX);
     KY = ifftshift(KY);
     KZ = ifftshift(KZ);
  
    % 3 x 1 x N x N x N
    k =  permute(cat(4, KX, KY, KZ),[4,5,1,2,3]);

    % k^2
    k2 = KX.*KX + KY.*KY + KZ.*KZ;
    clear KX KY KZ
    
    % Create Geomerty and Apply tensor at each space-point
    [ChiMap,Geom,X,Y] = Create_Geom_with_Tensor(mSize,vSize,Rin,Rout,IsoChi,AniChi);
    
   
  
    %off-resonce frequency equation
    
    %DiOmega = gyro * B0 * real(ifftn((1/3 * BDir' * fftnTensor(ChiMap) * BDir) - ((kz * k' * fftnTensor(ChiMap) * BDir / k2)));
    
    % first part
    
        % B'*ChiMap
        a1 = squeeze(mtimesx(BDir,'c',ChiMap));
 
        % 1/3 *B'*ChiMap*B
        a2 = 1/3 .* squeeze(mtimesx(a1,'c',BDir));
        
        FFTa2 = fftn(a2);
    
    % second part
    
        %B-projection in k-space:    B'*k
        kz = squeeze(mtimesx(BDir,'c',k));

      
        % ChiMap*B    3xNxNxN
        a3 = squeeze(mtimesx(ChiMap,'c',BDir));
        
        %  FTT(ChiMap*B)    
        FFTa3 = fftnTensor(a3);
        
        % -> 3x1xNxNxN
        FFTa3 = permute(FFTa3,[1,5,2,3,4]);
        
        % k'* FTT(ChiMap*B)
        a4 = squeeze(mtimesx(k,'c',FFTa3));
    

        % 1/3*B'*ChiMap*B - B'*k * k'*FFT(chiMap*B) / k2
        tmp = (kz .* a4 ./ k2);
        
        tmp = FFTa2 - tmp;
        tmp(k2 == 0) = 0;
        
      
%     DiOmega = gyro * B0 * real(ifftn(tmp));
     DiOmega = gyro * B0 * real(ifftn(FFTa2));
end

function [ChiMap,Geom,X,Y] = Create_Geom_with_Tensor(mSize,vSize,Rin,Rout,IsoChi,AniChi)

 [X, Y, ~] = ndgrid(linspacePeriodic(-vSize(1)/2,vSize(1)/2, mSize(1)),...
                    linspacePeriodic(-vSize(2)/2,vSize(2)/2, mSize(2)),...
                    linspacePeriodic(-vSize(3)/2,vSize(3)/2, mSize(3)));
      

  
   
    Geom = ((Rin^2 < X.^2 + Y.^2) & (X.^2 + Y.^2 < Rout^2));
    
%     Geom = randn(mSize.')>0;
%     [X, Y, Z] = ndgrid([1:mSize(1)],...
%                        [1:mSize(2)],...
%                        [1:mSize(3)]);
     

  
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

ChiT = [cos(fi), -sin(fi), 0;
       sin(fi), cos(fi), 0;
              0,       0, 1] * ...
       [IsoChi+AniChi,                0,                0;
                    0, IsoChi-AniChi./2,                0;
                    0,                0, IsoChi-AniChi./2] * ...
       [cos(fi), sin(fi), 0;
        -sin(fi),  cos(fi), 0;
              0,        0, 1]; 

% ChiT = IsoChi     .* [1,   0,   0;
%                       0,   1,   0;
%                       0,   0,   1] +...
%        AniChi*3/4 .* [cos(2*fi)+1/3,         -sin(2*fi),             0;
%                       -sin(2*fi),            1/3 - cos(2*fi),        0;
%                       0,                                   0,     -2/3];
%      




end


function [FFTTensor] = fftnTensor(a3)

FFTTensor = a3;
for ii = 1:3
    FFTTensor(ii,:,:,:) = fftn(a3(ii,:,:,:));
end

end


function [DiOmega,Geom,X,Y] = runMinExample

mSize = [200,200,200];
vSize = [1,1,1];
siz = [100,100,100];
dxyz = [0.01,0.01,0.01];
alpha = pi/2;
gyro = 2.67515255e8;  % Gyromagnetic ratio [rad/(T*s)]
B0 = -3.0;
Rin = 0.08;
Rout = 0.1;
%IsoChi = -60e-9;   % isotropic magnetic susceptibility
IsoChi = 0;
AniChi = -120e-9;  % anisotropic magnetic susceptibility
%AniChi = 0;  

[DiOmega,Geom,X,Y] = calculate_anisotropic_omega(mSize,vSize,alpha,gyro,B0,Rin,Rout,IsoChi,AniChi);

end
