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

% Anonymous functions and parameters for minimization
get_Vmap    =  @(R,p0) getCylinderMask( [G.GridSize(1:2),1], G.SubVoxSize, G.VoxelCenter, [G.VoxelSize(1:2),G.SubVoxSize], p0, G.vz0, R*ones(1,G.Nmajor), G.vx0, G.vy0, isUnit, isCentered, prec, [] );
get_BV      =  @(R,p0) G.GridSize(3) * sum(reshape(get_Vmap(R,p0),[],1));
get_rand_p0 =  @() G.p0 + G.SubVoxSize * [2*rand(2,G.Nmajor)-1; zeros(1,G.Nmajor)];
dBV_Fun     =  @(R) 2*pi*R * G.Nmajor * (G.GridSize(3) / G.SubVoxSize^2);
r0_bounds   =  [ 0.5 * G.Rmajor, min( 1.5 * G.Rmajor, min(G.VoxelSize(1:2))/(2*sqrt(G.Nmajor)) ) ]; % max allowed radius satisfies 2*R*sqrt(N) = Width, e.g. if they were all in a regular grid
BV_Target   =  G.Targets.aBVF * prod(G.GridSize); % Blood Volume in units of voxels
BV_Tol      =  0.001 * BV_Target * (G.SubVoxSize/max(G.r0));

% NOTE: fzero tends to take too long to converge by finding the exact zero
% fzero_opts  =  optimset('TolX',1e-8*G.SubVoxSize);%,'PlotFcns',{@optimplotx,@optimplotfval});

% NOTE: must use a minimization that allows bound constraints
% NOTE: Matlab changed options names between versions, so this is a hack to
%       get around that...
% R2016a_And_Later_Opts = struct(...
%     'Display',                  'final', ...
%     'FiniteDifferenceStepSize', 5*G.SubVoxSize, ...
%     'FiniteDifferenceType',     'central', ...
%     'FunctionTolerance',        1e-12 * BV_Target, ...
%     'StepTolerance',            1e-3 * G.SubVoxSize, ...
%     'TypicalX',                 mean(G.r0) ...
%     );
% R2015a_And_Earlier_Opts = struct(...
%     'Display',          'final', ...
%     'FinDiffRelStep',   5*G.SubVoxSize/mean(G.r0), ... % deltaR = FinDiffRelStep .* max(abs(R),TypicalX)
%     'FinDiffType',     	'central', ...
%     'TolFun',           1e-12 * BV_Target, ...
%     'TolX',             1e-3 * G.SubVoxSize, ...
%     'TypicalX',        	mean(G.r0) ...
%     );
% 
% f2016a = fields(R2016a_And_Later_Opts);
% f2015a = fields(R2015a_And_Earlier_Opts);
% lsqnonlin_opts = optimoptions('lsqnonlin');
% for ii = 1:max(length(f2016a),length(f2015a)) % should be same length
%     if any(ismember(fields(lsqnonlin_opts),f2016a{ii}))
%         lsqnonlin_opts.(f2016a{ii}) = R2016a_And_Later_Opts.(f2016a{ii});
%     elseif any(ismember(fields(lsqnonlin_opts),f2015a{ii}))
%         lsqnonlin_opts.(f2015a{ii}) = R2015a_And_Earlier_Opts.(f2015a{ii});
%     end        
% end

% Initial parameters
p0_CurrentBest   =  G.p0;
r0_CurrentBest   =  G.Rmajor;
BV_CurrentBest   =  get_BV(r0_CurrentBest, p0_CurrentBest);

iter = 0;
iter_MAX = 5;
while (iter < iter_MAX) && (abs(BV_CurrentBest - BV_Target) > BV_Tol)
    
    iter = iter + 1;
    
    p0 = get_rand_p0();
    
    %[r0, BV_err] = fzero( @(R) get_BV(R,p0) - BV_Target, r0_bounds, fzero_opts );
    %[r0, BV_err_norm, BV_err] = lsqnonlin( @fopt, G.Rmajor, ...
    %    r0 - 2*G.SubVoxSize, r0 + 2*G.SubVoxSize, lsqnonlin_opts );
    
    ri = linspace(r0_bounds(1),r0_bounds(2),9);
    yi = fopt(ri);
    cp = polyfit(ri,yi,2); % Should be approx. quandratic; minimize this first
    fp = @(R) polyval(cp,R);
    r0 = fzero(fp,r0_bounds);
    
    foptsq_penalized = @(R) (fopt(R)/prod(G.GridSize)).^2 + 1e-9*((R-r0)/r0).^2;
    [r0,BV_err_norm] = patternsearch(foptsq_penalized,r0,[],[],[],[],r0-0.5*G.SubVoxSize,r0+0.5*G.SubVoxSize);
    BV_err = sqrt(BV_err_norm); %unsigned, but doesn't matter
    
    if abs(BV_err) < abs(BV_CurrentBest - BV_Target)
        p0_CurrentBest = p0;
        r0_CurrentBest = r0;
        BV_CurrentBest = get_BV(r0,p0);
    end
    
    G.Rmajor = r0_CurrentBest;
    G.aBVF = BV_CurrentBest/prod(G.GridSize);
    G.p0 = p0_CurrentBest;
    if prnt; G.ShowBVFResults; end
    
end

    function y = fopt(R)
        y = zeros(size(R));
        for idx = 1:numel(R)
            y(idx) = get_BV(R(idx),p0) - BV_Target;
        end
    end

if iter >= iter_MAX
    warning('Number of iterations exceeded for: Major Vasculature');
end

G.Rmajor = r0_CurrentBest;
G.aBVF = BV_CurrentBest/prod(G.GridSize);
G.p0 = p0_CurrentBest;

G.RmajorFun = @(varargin) G.Rmajor*ones(varargin{:});
G.r0 = G.RmajorFun(1,G.Nmajor);

end

function G = ImproveMajorBVF_legacy(G)

% Ensure cylinder direction vectors are unit, and set parameters
G           =   NormalizeCylinderVecs(G);
isUnit      =   true;
isCentered    =   true;
prec        =   'double';

% Calculate Map for Major Vessels
R0                  =   G.Rmajor;
BV_Best             =   G.Targets.aBVF * prod(G.GridSize); % Blood Volume in units of voxels
VMap_RmajorSlice    =    @(R,p0) getCylinderMask( [G.GridSize(1:2),1], G.SubVoxSize, G.VoxelCenter, [G.VoxelSize(1:2),G.SubVoxSize], ...
    p0, G.vz0, R*ones(1,G.Nmajor), G.vx0, G.vy0, isUnit, isCentered, prec, [] );
get_BV_RmajorSlice  =   @(R,p0) G.GridSize(3) * sum(reshape(VMap_RmajorSlice(R,p0),[],1));
% VMap_Rmajor       =   @(R,p0) getCylinderMask( G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
%     G.p0, vz0, R*ones(1,G.Nmajor), vx0, vy0, isUnit, isCentered, prec, [] );
% get_BV_Rmajor     =   @(R,p0) sum(reshape(VMap_Rmajor(R,G.p0),[],1));

BV_CurrentBest   = -inf;
p0_CurrentBest   =  G.p0;
R_CurrentBest    =  R0 * ones(1,G.Nmajor);

iters = 0;
iters_MAX = 50;
while (iters < iters_MAX) && (abs(BV_CurrentBest - BV_Best) > 0.25 * (1/max(G.GridSize)) * BV_Best)
    
    iters = iters + 1;
    
    p1 = G.p0;
    if iters > 1
        p1 = p1 + G.SubVoxSize * [2*rand(2,size(G.p0,2))-1; zeros(1,size(G.p0,2))];
    end
    
    [Rlow,Rhigh] = deal( 0.5 * R0, 1.5 * R0 );
    %[BVlow,BVhigh] = deal( get_BV_Rmajor(Rlow,p1), get_BV_Rmajor(Rhigh,p1) );
    [BVlow,BVhigh] = deal( get_BV_RmajorSlice(Rlow,p1), get_BV_RmajorSlice(Rhigh,p1) );
    
    [BVlow_last,BVhigh_last] = deal(-1);
    
    while ~(BVlow == BVlow_last && BVhigh == BVhigh_last)
        [BVlow_last,BVhigh_last] = deal(BVlow,BVhigh);
        
        Rmid    =   (Rlow + Rhigh)/2;
        %BVmid    =   get_BV_Rmajor(Rmid,p1);
        BVmid    =   get_BV_RmajorSlice(Rmid,p1);
        
        if BVmid >= BV_Best
            BVhigh    =   BVmid;
            Rhigh    =   Rmid;
        else
            BVlow    =   BVmid;
            Rlow    =   Rmid;
        end
    end
        
    if abs(BVhigh - BV_Best) <= abs(BVlow - BV_Best)
        BVmid = BVhigh;
        Rmid  = Rhigh;
    else
        BVmid = BVlow;
        Rmid  = Rlow;
    end
    
    if abs(BVmid - BV_Best) < abs(BV_CurrentBest - BV_Best)
        BV_CurrentBest = BVmid;
        R_CurrentBest  = Rmid;
        p0_CurrentBest = p1;
    end
    
end
if iters >= iters_MAX
    warning('Number of iterations exceeded for: Major Vasculature');
end

G.Rmajor = R_CurrentBest;
G.aBVF = BV_CurrentBest/prod(G.GridSize);
G.p0 = p0_CurrentBest;

G.RmajorFun = @(varargin) G.Rmajor*ones(varargin{:});
G.r0 = G.RmajorFun(1,G.Nmajor);

end