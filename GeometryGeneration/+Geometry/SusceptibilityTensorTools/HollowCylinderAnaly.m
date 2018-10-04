function [AnalAniDiOmega] = HollowCylinderAnaly(Geom,X,Y,Rin,Rout,alpha,IsoChi,AniChi,gyro,B0)

r = sqrt(X.*X + Y.*Y);
fi = atan2(Y,X);

c2 = cos(alpha)^2;
s2 = sin(alpha)^2;

% Rin<r<Rout
AnalAniDiOmega = Geom .* gyro*B0 .* ( IsoChi/2 .* ( c2 - 1/3 - s2 * cos(2*fi) .* (Rin^2./r.^2) ) + ... 
                 AniChi .* (s2 .* (-5/12 - (cos(2*fi)/8) .* (1 + Rin^2./r.^2) + (3/4) * log(Rout./r) ) - c2/6 ) );
% r>Rout            
AnalAniDiOmega = AnalAniDiOmega + (~Geom & (r>(Rin+Rout)/2)) .* ( gyro*B0) .* ...
                     (IsoChi .* ( (s2 * cos(2*fi))/2 .* ((Rout^2-Rin^2)./r.^2) ) + ...
                      AniChi .* ( (s2 * cos(2*fi))/8 .* ((Rout^2-Rin^2)./r.^2) ) );
% r<Rin               
%AnalAniDiOmega(~Geom & (r<(Rin+Rout)/2)) = 0 + gyro*B0*AniChi * (3 * s2 / 4) * (log(Rout/Rin));
AnalAniDiOmega = AnalAniDiOmega + (~Geom & (r<(Rin+Rout)/2)) .* ...
                    ( 0 + gyro*B0*AniChi * (3 * s2 / 4) * (log(Rout/Rin)));

end
