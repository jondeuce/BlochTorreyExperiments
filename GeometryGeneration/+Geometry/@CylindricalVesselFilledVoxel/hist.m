function [ ] = hist( Geom )
%HIST(Geom) Plots a histogram of minor voxel angles vs. bin counts.

v  = Geom.vz;
vz = [zeros(2,size(v,2));ones(1,size(v,2))];
th = atan2( sqrt( sum( cross( v, vz, 1 ).^2, 1 ) ), abs( dot( v, vz, 1 ) ) );
nth = cellfun(@numel,Geom.idx);

% nbins = min(15, floor(numel(th)/5));
% figure
% title('$\theta$')
% hist(th,nbins);

figure, plot(th, nth, '*');

end

