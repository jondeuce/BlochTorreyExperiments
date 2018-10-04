function [DiOmega, Geom] = SeparateOmega(mSize,IsoChi,gyro,B0)
%SEPARATEOMEGA Calculates the off-resonance frequency for different
% orientation and cylinder radious

    if nargin == 0
        [DiOmega, Geom] = runMinExample;
        return
    end

    [DiOmega, Geom] = calculate_anisotropic_omega(mSize,IsoChi,gyro,B0);
    
end

function [DiOmega, Geom] = calculate_anisotropic_omega(mSize,IsoChi,gyro,B0)
   %mSize: grid,  vSize: unitsTensorMap = zeros(ChiMapSize);
    
   %BDir  = [sin(alpha); 0; cos(alpha)];
   BDir = [1;0;0];
   
   
   mSize = mSize(:);
    
    % Create Geomerty and Apply tensor at each space-point
    [ChiMap, Geom] = Create_Geom_with_Tensor(mSize,IsoChi);
    
    % FFT of tensor map 3x3xNxNxN
    
    
     %DiOmega = gyro * B0 * real(ifftn((1/3 * BDir' * fftnTensor(ChiMap) * BDir);
    
     % first part
    
        
        % ChiMap' * B
        a1 = squeeze(mtimesx(ChiMap,'c',BDir));
        
        FFTa1 = fftnTensor(a1); 
 
        % 1/3 *B'*ChiMap * B
        Apart = 1/3 .* squeeze(mtimesx(BDir,'c',FFTa1));
    
        
     %DiOmega = gyro * B0 * real(ifftshift(ifftn(Apart)));
     DiOmega = gyro * B0 * real(ifftn(Apart));
     
end

function [ChiMap, Geom] = Create_Geom_with_Tensor(mSize,IsoChi)



% Geom = ( abs(X) < mSize(1)/4 & abs(Y) < mSize(2)/4 & abs(Z) < mSize(3)/4 ) & ...
%       ~( abs(X) < mSize(1)/8 & abs(Y) < mSizeS(2)/8 & abs(Z) < mSize(3)/8 ) ;

Geom = randn(mSize.')>0;

ChiMap = ApplyTensor(Geom,mSize,IsoChi);

end

function [TensorMap] = ApplyTensor(Geom,mSize,IsoChi)
% Map is NxNxN

ChiMapSize = [3,3,size(Geom)];
TensorMap = zeros(ChiMapSize);

for ii = 1:mSize(1)
    for jj = 1:mSize(2)
        for kk = 1:mSize(3)
            %fi = atan2(Y(ii,jj,kk),X(ii,jj,kk));
            % apply susceptibility only on cylinder points
            if Geom(ii,jj,kk) == 1
                TensorMap(:,:,ii,jj,kk) = IsoChi.*eye(3);
            else 
                TensorMap(:,:,ii,jj,kk) = zeros(3);
            end
        end
    end    
end

end


function ChiT = Tensor(IsoChi)

ChiT = [IsoChi,                0,                0;
                    0, IsoChi,                0;
                    0,                0, IsoChi]; 

end

function [Y] = fftnTensor(a1)

Y = a1;
 for ii = 1:3
      %Y(ii,:,:,:) = fftn(fftshift(a1(ii,:,:,:)));
      Y(ii,:,:,:) = fftn(a1(ii,:,:,:));
 end

end

function [DiOmega,Geom] = runMinExample

mSize = [5,5,5];
vSize = [1,1,1];
alpha = pi/2;
gyro = 2.67515255e8;  % Gyromagnetic ratio [rad/(T*s)]
B0 = -3.0;
CylinderRad = 0.35;
IsoChi = -60e-9;   % isotropic magnetic susceptibility
%AniChi = -120e-9;  % anisotropic magnetic susceptibility
AniChi = 0;  

[DiOmega, Geom] = calculate_anisotropic_omega(mSize,gyro,B0,IsoChi);

end