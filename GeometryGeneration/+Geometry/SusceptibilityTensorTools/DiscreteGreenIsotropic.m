function [OmegaDiscrt,X,Y,Z,Geom,DiscreteGreen] = DiscreteGreenIsotropic(gyro,B0,alpha,IsoChi,siz,dxyz,Rout)

BDir  = [sin(alpha); 0; cos(alpha)];

mLow  = floor(siz / 2);
mHigh = mLow - ~mod(siz, 2);

mLow  = mLow  .* dxyz;
mHigh = mHigh .* dxyz;

[X, Y, Z] = ndgrid(-mLow(1) : dxyz(1) : mHigh(1), ...
                    -mLow(2) : dxyz(2) : mHigh(2), ...
                    -mLow(3) : dxyz(3) : mHigh(3));

Geom = (X.^2 + Y.^2 < Rout^2);        

[sz1,sz2,sz3] = size(Geom);
OmegaDiscrt = zeros(sz1,sz2,sz3);
r = zeros(sz1,sz2,sz3);
DiscreteGreen = zeros(sz1,sz2,sz3);

%r = sqrt(X.*X + Y.*Y + Z.*Z);
r2 = X.*X + Y.*Y + Z.*Z;
rz = X.*BDir(1) + Y.*BDir(2) + Z.*BDir(3);

DiscreteGreen = (3*(rz.*rz) - r2)./(4*pi*sqrt(r2.*r2.*r2.*r2.*r2));
DiscreteGreen(r2==0) = 0; 
   
OmegaDiscrt = gyro*B0*IsoChi*real(ifftn(fftn(Geom).*fftn(ifftshift(DiscreteGreen))));

end