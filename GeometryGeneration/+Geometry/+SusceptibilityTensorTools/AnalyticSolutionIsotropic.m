function [AnalDiOmega, AnalBz] = AnalyticSolutionIsotropic(Geom,X,Y,CylinderRad,alpha,IsoChi,gyro,B0)
% Analytical solution for cylinder with Radious: CylinderRad

[sz1,sz2,sz3] = size(Geom);
AnalDiOmega = zeros(sz1,sz2,sz3);
AnalBz = zeros(sz1,sz2,sz3);
for ii = 1:sz1
    for jj = 1:sz2
        for kk = 1:sz3
            if Geom(ii,jj,kk)
                  AnalDiOmega(ii,jj,kk) = gyro*IsoChi*B0*(3*cos(alpha)^2-1)/6;
                  AnalBz(ii,jj,kk) = IsoChi*B0*(3*cos(alpha)^2-1)/6;
            else
                AnalDiOmega(ii,jj,kk) = (gyro*IsoChi*B0*(sin(alpha)^2)*(CylinderRad^2)*...
                    ((X(ii,jj,kk)^2)-(Y(ii,jj,kk)^2)))/(2*((Y(ii,jj,kk)^2)+(X(ii,jj,kk)^2))^2);
                AnalBz(ii,jj,kk) = (IsoChi*B0*(sin(alpha)^2)*(CylinderRad^2)*...
                    ((X(ii,jj,kk)^2)-(Y(ii,jj,kk)^2)))/(2*((Y(ii,jj,kk)^2)+(X(ii,jj,kk)^2))^2);
            end
        end
    end
end
                       
end

  