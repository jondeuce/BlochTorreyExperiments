%% Box dimensions
GridSize = [512/4, 512/2, 512];
BoxDims = [3000/4, 3000/2, 3000];
BoxCenter = [0,0,0];

%% Cylinders
N = 2000;
r = 25.0 * ones(1,N);
% [p, v, r] = sampleRandomCylinders( BoxDims, BoxCenter, r, N, 'PointAndDirection' );
% [p, v, r] = sampleRandomCylinders( BoxDims, BoxCenter, r, N, 'BoundingSpherePair' );
[p, v, r] = sampleRandomCylinders( BoxDims, BoxCenter, r, N, 'BoundingSphereAndDirection' );

%% Angular distribution
getMaskFun = @(p,v,r) getCylinderMask( ...
    GridSize, BoxDims./GridSize, BoxCenter, BoxDims, ...
    p, v, r, [], [], true, true);

th = acos(v(3,:)); % polar angle w.r.t. z-axis
phi = atan2(v(2,:), v(1,:)); % azimuthal angle in xy-plane
counts = zeros(size(th));
th_rep = cell(size(th));
phi_rep = cell(size(phi));
for ii = 1:length(r)
    b = getMaskFun(p(:,ii), v(:,ii), r(ii));
    counts(ii) = sum(b(:));
    th_rep{ii} = th(ii)*ones(1, counts(ii));
    phi_rep{ii} = phi(ii)*ones(1, counts(ii));
end

%% Plot theta
th_all = cat(2, th_rep{:});
figure, histogram(th_all, 20, 'normalization', 'pdf');
hold on; fplot(@sin, [0,pi/2], 'r--'); hold off

%% Plot phi
phi_all = cat(2, phi_rep{:});
figure, histogram(phi_all, 50, 'normalization', 'pdf');
hold on; fplot(@(x) ones(size(x))/(2*pi), [-pi,pi], 'r--'); hold off