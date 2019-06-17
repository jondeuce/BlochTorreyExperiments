function [ G ] = ImproveMajorBVF( G, prnt )
%IMPROVEMAJORBVF

if nargin < 2
    prnt = false;
end

% Ensure cylinder direction vectors are unit, and set parameters
G           =   NormalizeCylinderVecs(G);
isUnit      =   true;
isCentered  =   true;
prec        =   'double';

% MAJOR_GAP   =   max(G.VoxelSize)/sqrt(G.Nmajor) - 2 * max(G.r0); % Approx. average distance between major vessels
MAJOR_GAP   =   min(G.VoxelSize); % for Nmajor = 1 case
for ii = 2:G.Nmajor
    for jj = 1:ii-1
        MAJOR_GAP = min(MAJOR_GAP, norm(G.p0(:,ii) - G.p0(:,jj)));
    end
end
MAJOR_GAP   =   MAJOR_GAP - 2*G.Rmajor;
P0_FACT     =   0.03 * MAJOR_GAP / G.SubVoxSize; % Allow vessels to move about 3% from where they started

% We call the angle of the major vessels to be effectively zero if the sine
% of the angle is less than one tenth the smallest relative subvoxel size,
% as in this case, the angle between the major vessel and vertical would
% not affect the discrete cylinder map significantly, and we can
% approximate the cylinders as uniformly vertical.
% Similarly, we can check if the cosine of the angle is small to check if
% the vessel is effectively perpendicular.
isAngleZero = @(th) abs(sind(th)) < 0.1 * G.SubVoxSize / max(G.VoxelSize);
isAngleNinety = @(th) abs(cosd(th)) < 0.1 * G.SubVoxSize / max(G.VoxelSize);

% Anonymous functions and parameters for minimization
if isAngleZero( G.MajorAngle )
    SliceGsize  =  [G.GridSize(1:2), 1];
    SliceVsize  =  [G.VoxelSize(1:2), G.SubVoxSize];
    RepDimGsize =  G.GridSize(3); % repeated dimension size
    
    get_Vmap    =  @(R,p0) getCylinderMask( SliceGsize, G.SubVoxSize, G.VoxelCenter, SliceVsize, p0, G.vz0, R*ones(1,G.Nmajor), G.vx0, G.vy0, isUnit, isCentered, prec, [] );
    get_BV      =  @(R,p0) RepDimGsize * sum(vec(get_Vmap(R,p0)));
    get_rand_p0 =  @() G.p0 + P0_FACT * G.SubVoxSize * [2*rand(2,G.Nmajor)-1; zeros(1,G.Nmajor)];
    % dBV_Fun     =  @(R) 2*pi*R * G.Nmajor * (RepDimGsize / G.SubVoxSize^2);
    
    % max allowed radius satisfies 2*R*sqrt(N) = Width, e.g. if they were all in a regular grid
    r0_bounds   =  [ 0.5 * G.Rmajor, min( 1.5 * G.Rmajor, min(G.VoxelSize(1:2))/(2*sqrt(G.Nmajor)) ) ];
elseif isAngleNinety( G.MajorAngle )
    SliceGsize  =  [1, G.GridSize(2:3)];
    SliceVsize  =  [G.SubVoxSize, G.VoxelSize(2:3)];
    RepDimGsize =  G.GridSize(1); % repeated dimension size
    
    get_Vmap    =  @(R,p0) getCylinderMask( SliceGsize, G.SubVoxSize, G.VoxelCenter, SliceVsize, p0, G.vz0, R*ones(1,G.Nmajor), G.vx0, G.vy0, isUnit, isCentered, prec, [] );
    get_BV      =  @(R,p0) RepDimGsize * sum(vec(get_Vmap(R,p0)));
    get_rand_p0 =  @() G.p0 + P0_FACT * G.SubVoxSize * [zeros(1,G.Nmajor); 2*rand(2,G.Nmajor)-1];
    % dBV_Fun     =  @(R) 2*pi*R * G.Nmajor * (RepDimGsize / G.SubVoxSize^2);
    
    % max allowed radius satisfies 2*R*sqrt(N) = Width, e.g. if they were all in a regular grid
    r0_bounds   =  [ 0.5 * G.Rmajor, min( 1.5 * G.Rmajor, min(G.VoxelSize(2:3))/(2*sqrt(G.Nmajor)) ) ];
else
    get_Vmap    =  @(R,p0) getCylinderMask( G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, p0, G.vz0, R*ones(1,G.Nmajor), G.vx0, G.vy0, isUnit, isCentered, prec, [] );
    get_BV      =  @(R,p0) sum(vec(get_Vmap(R,p0)));
    get_rand_p0 =  @() G.p0 + roty(G.MajorAngle) * [P0_FACT * G.SubVoxSize * (2*rand(2,G.Nmajor)-1); zeros(1,G.Nmajor)];
    
    % TODO: this has not been updated
    % dBV_Fun   =  @(R) 2*pi*R * G.Nmajor * (G.GridSize(3) / G.SubVoxSize^2);
    
    % max allowed radius satisfies 2*R*sqrt(N) = Width, e.g. if they were all in a regular grid
    r0_bounds   =  [ 0.5 * G.Rmajor, min( 1.5 * G.Rmajor, min(G.VoxelSize)/(2*sqrt(G.Nmajor)) ) ];
end

BV_Target   =  G.Targets.aBVF * prod(G.GridSize); % Blood Volume in units of voxels
BV_Tol      =  0.0001 * BV_Target * (G.SubVoxSize/max(G.r0));

% Initial parameters
p0_CurrentBest   =  G.p0;
r0_CurrentBest   =  G.Rmajor;
BV_CurrentBest   =  get_BV(r0_CurrentBest, p0_CurrentBest);

iter = 0;
iter_MAX = 10;
while (iter < iter_MAX) && (abs(BV_CurrentBest - BV_Target) > BV_Tol)
    
    iter = iter + 1;
    
    p0 = get_rand_p0();
    
    % ==== Using fzero ==== %
    fzero_opts = optimset('TolX', 1e-6 * G.SubVoxSize); % use low tolerance, but lots of iterations
    [r0, BV_err] = fzero(@fopt, r0_bounds, fzero_opts);
    
    % ==== Using lsqnonlin ==== %
    %     [r0, BV_err_norm, BV_err] = lsqnonlin( @fopt, G.Rmajor, r0 - 2*G.SubVoxSize, r0 + 2*G.SubVoxSize, lsqnonlin_opts );
    
    % ==== Using patternsearch ==== %
    %     ri = linspace(r0_bounds(1),r0_bounds(2),5);
    %     yi = fopt(ri);
    %     cp = polyfit(ri,yi,2); % Should be approx. quandratic; minimize this first
    %     fp = @(R) polyval(cp,R);
    %     r0 = fzero(fp,r0_bounds);
    %
    %     % minimize objective with small penalty on moving far from initial
    %     % guess with the hopes of providing a unique solution
    %     foptsq_penalized = @(R) (fopt(R)/prod(G.GridSize)).^2 + 1e-9*((R-r0)/r0).^2;
    %
    %     [Acon,bcon,Aeq,beq] = deal([]);
    %     [lb,ub] = deal(r0 - 0.5*G.SubVoxSize, r0 + 0.5*G.SubVoxSize);
    %     [r0,BV_err_norm] = patternsearch(foptsq_penalized,r0,Acon,bcon,Aeq,beq,lb,ub,...
    %         ...%psoptimset('tolx', 1e-6*G.SubVoxSize, 'tolmesh', 1e-6*G.SubVoxSize));
    %         psoptimset('tolx', 1e-6 * max(G.r0), 'tolmesh', 1e-6 * max(G.r0)) );
    %     BV_err = sqrt(BV_err_norm); %unsigned, but doesn't matter
    
    if abs(BV_err) < abs(BV_CurrentBest - BV_Target)
        p0_CurrentBest = p0;
        r0_CurrentBest = r0;
        BV_CurrentBest = get_BV(r0,p0);
    end
    
    G.Rmajor = r0_CurrentBest;
    G.aBVF = BV_CurrentBest/prod(G.GridSize);
    G.p0 = p0_CurrentBest;
    
    %     if prnt
    %         G.ShowBVFResults
    %     end
    
end

    function y = fopt(R)
        y = zeros(size(R));
        for idx = 1:numel(R)
            y(idx) = get_BV(R(idx),p0) - BV_Target;
        end
    end

% if iter >= iter_MAX
%     warning('Number of iterations exceeded for: Major Vasculature');
% end
if prnt
    G.ShowBVFResults
end

Rmajor = r0_CurrentBest;
G.Rmajor = Rmajor;
G.aBVF = BV_CurrentBest/prod(G.GridSize);
G.p0 = p0_CurrentBest;

G.RmajorFun = @(varargin) Rmajor .* ones(varargin{:});
G.r0 = G.RmajorFun(1, G.Nmajor);

end
