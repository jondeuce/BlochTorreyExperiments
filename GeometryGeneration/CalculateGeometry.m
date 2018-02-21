function [ Geometry ] = CalculateGeometry( Params, SimSettings )
%CALCULATEGEOMETRY Calculates geometry for simulation

switch SimSettings.Dimension
    case 2
        Geometry	=   CalculateGeometry_2D( Params, SimSettings );
    case 3
        Geometry	=   CalculateGeometry_3D( Params, SimSettings );
end

end

function Geometry = CalculateGeometry_2D( Params, SimSettings )

%==========================================================================
% Parse settings
%==========================================================================
n	=	SimSettings.NumMajorVessels; % number of major vessels
N	=	Params.NumMinorVessels; % number of minor vessels

%==========================================================================
% Extract commonly used parameters for conciseness
%==========================================================================
PlotGeo     =   SimSettings.flags.PlotAnything && SimSettings.flags.PlotGeometry;
VoxelSize	=   SimSettings.VoxelSize;
SubVoxSize	=   SimSettings.SubVoxSize;
VoxelCenter	=   SimSettings.VoxelCenter;
GridSize	=   SimSettings.GridSize;
RepStr      =   num2str( Params.Rep );

%==========================================================================
% Calculate Cylinders
%==========================================================================
%   p [3 x n+N] is a point on the cylinder axis
%   v [3 x n+N] is the axis direction
%   r [1 x n+N] is the cylinder radius
t_Cylinders = tic;

[x,y]	=	regular_grid_2D(n,false);
p0      =   [ VoxelSize(1) * x(:)'; VoxelSize(2) * y(:)'; zeros(1,n,'double') ];
p0      =   bsxfun( @plus, VoxelCenter(:), p0 );
v0      =	repmat( double([0;0;1]), [1,n] );
r0      =	repmat( double(Params.R_Major), [1,n] );

[p,r,vz]   	=   deal(p0,r0,v0);
[vx,vy,vz]	=   nullVectors3D(vz);

% Find best major vessel radii
R0              =   Params.R_Major;
BV_Best         =   round( (Params.Major_BloodVol/GridSize(3)) * (prod(GridSize)/SimSettings.Total_Volume) );
VMap_Rmajor     =	@(R) getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
    p, vz, R*ones(1,SimSettings.NumMajorVessels), vx, vy, true, true, 'double', [], [], true );
get_BV_Rmajor	=   @(R) sum(reshape(VMap_Rmajor(R),[],1));

[Rlow,Rhigh]	=   deal( 0.5 * R0, 1.5 * R0 );
[BVlow,BVhigh]	=   deal( get_BV_Rmajor(Rlow), get_BV_Rmajor(Rhigh) );

[BVlow_last,BVhigh_last]	=   deal(-1);

while ~(BVlow == BVlow_last && BVhigh == BVhigh_last)
    [BVlow_last,BVhigh_last]	=   deal(BVlow,BVhigh);
    
    Rmid	=   (Rlow + Rhigh)/2;
    BVmid	=   get_BV_Rmajor(Rmid);
    
    if BVmid >= BV_Best
        BVhigh	=   BVmid;
        Rhigh	=   Rmid;
    else
        BVlow	=   BVmid;
        Rlow	=   Rmid;
    end
end

if abs(BVhigh - BV_Best) <= abs(BVlow - BV_Best)
    r	=   Rhigh * ones(1,n);
else
    r	=   Rlow * ones(1,n);
end

VasculatureMap	=   [];
mx              =   cell(1,n);
bx              =   zeros(3,8,n);
for ii = 1:n
    [ VasculatureMap, mx{ii} ]	=	getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
        p(:,ii), vz(:,ii), r(ii), vx(:,ii), vy(:,ii), true, true, 'double', VasculatureMap, [], true );
    bx(:,:,ii)	=	cylinderInBox2OOBB( p(:,ii), vz(:,ii), r(ii), VoxelSize, VoxelCenter );
end

% Add minor vasculature
BVF         =	@(x) sum(x(:))/numel(x);
BVF_Major	=   BVF(VasculatureMap);
BVF_Minor	=   0;
idx         =   n;
[p,vz,r,mx,bx]	=	deal([p,zeros(3,N)],[vz,zeros(3,N)],[r,zeros(1,N)],[mx,cell(1,N)],cat(3,bx,zeros(3,8,N)));

% Normal Random Vectors
StandardVec	=   @(v) (v*mysign(v(3)))/norm(v); %Normalize Vector and make z-component positive
NormRandVec	=   @() randn(3,1);

% Discrete Random Vectors
NumBins    	=   5;
Theta      	=   acos(fliplr(linspaceVec(0,1,NumBins,[1,0]))); % Angles in [0,pi/2)
Phi        	=   2*pi*linspaceVec(0,1,2*NumBins,[0,1]); % Angles in [0,2*pi)
SphereVec	=   @(theta,phi) [ cos(phi)*sin(theta); sin(phi)*sin(theta); cos(theta) ];

% GetNewVec	=   @() StandardVec(NormRandVec());
GetNewVec	=   @() SphereVec( Theta(randi(NumBins)), Phi(randi(2*NumBins)) );

while BVF_Major + BVF_Minor < Params.Total_BVF
    [x,y]	=	deal(rand-0.5,rand-0.5);
    p1      =   [VoxelSize(1) * x; VoxelSize(2) * y; 0] + VoxelCenter(:);
    v1      =	GetNewVec();
    [v1x,v1y,v1z]	=   nullVectors3D(v1);
    r1      =	SimSettings.R_Minor_mu + randn * SimSettings.R_Minor_sig;
    bx1     =   cylinderInBox2OOBB(p1,v1,r1,VoxelSize,VoxelCenter);
    
    %if ~any(cylinderIntersection(p1,v1z,r1,p(:,1:idx),vz(:,1:idx),r(1:idx)))
    if ~any(cylinderIntersectionInBox(p1,v1z,r1,bx1,p(:,1:idx),vz(:,1:idx),r(1:idx),bx(:,:,1:idx)))
        idx         =   idx + 1;
        p(:,idx)	=	p1;
        vz(:,idx)	=   v1z;
        r(:,idx)	=   r1;
        bx(:,:,idx)	=   bx1;
        [VasculatureMap,mx{idx}]	=   getCylinderMask(GridSize,SubVoxSize,VoxelCenter,VoxelSize,p1,v1z,r1,v1x,v1y,true,true,'double',VasculatureMap,[],true);
        BVF_Minor                   =   BVF(VasculatureMap) - BVF_Major;
    end
end
[p,vz,r,mx,bx]	=   deal(p(:,1:idx),vz(:,1:idx),r(1:idx),mx(1:idx),bx(:,:,1:idx));
[vx,vy,vz]	=   nullVectors3D(vz);

VmapLast	=	getCylinderMask(GridSize,SubVoxSize,VoxelCenter,VoxelSize,p(:,1:end-1),vz(:,1:end-1),r(1:end-1),vx(:,1:end-1),vy(:,1:end-1),true,true,'double',[],[],true);
if abs(BVF(VmapLast) - Params.Total_BVF) < abs(BVF(VasculatureMap) - Params.Total_BVF)
    VasculatureMap      =   VmapLast;
    [p,vz,r,vx,vy,mx]	=   deal(p(:,1:end-1),vz(:,1:end-1),r(1:end-1),vx(:,1:end-1),vy(:,1:end-1),mx(1:end-1));
end
clear VmapLast

t_Cylinders	=   toc(t_Cylinders);
display_toc_time( t_Cylinders, 'Generating Major/Minor Cylinders ' );

[ms,me,Ms,Me,Ts,Te]	=   deal( BVF_Minor, Params.MinorVessel_RelBVF * Params.Total_BVF, ...
    BVF_Major, (1.0-Params.MinorVessel_RelBVF) * Params.Total_BVF, ...
    BVF_Minor + BVF_Major, Params.Total_BVF );

relerr	=   @(x,y) max(abs(x-y)./max(abs(x),abs(y)));
fprintf( '\n' );
fprintf( 'Minor BVF simulated:\t%0.6f\n', ms );
fprintf( 'Minor BVF expected: \t%0.6f\n', me );
fprintf( 'Minor BVF rel-error:\t%0.6f\n', relerr(ms,me) );
fprintf( 'Major BVF simulated:\t%0.6f\n', Ms );
fprintf( 'Major BVF expected: \t%0.6f\n', Me );
fprintf( 'Major BVF rel-error:\t%0.6f\n', relerr(Ms,Me) );
fprintf( 'Total BVF simulated:\t%0.6f\n', Ts );
fprintf( 'Total BVF expected: \t%0.6f\n', Te );
fprintf( 'Total BVF rel-error:\t%0.6f\n', relerr(Ts,Te) );
fprintf( '\n' );

%==========================================================================
% Return Geometry struct
%==========================================================================

i1	=   1:SimSettings.NumMajorVessels;
i2	=   SimSettings.NumMajorVessels+1:length(r);

Geometry	=   struct( ...
    'MainCylinders',    struct('p',p(:,i1),'vx',vx(:,i1),'vy',vy(:,i1),'vz',vz(:,i1),'r',r(i1),'mx',{mx(i1)},'bx',bx(:,:,i1),'n',numel(i1)),	...
    'MinorCylinders',   struct('p',p(:,i2),'vx',vx(:,i2),'vy',vy(:,i2),'vz',vz(:,i2),'r',r(i2),'mx',{mx(i2)},'bx',bx(:,:,i2),'n',numel(i2)),	...
    'VasculatureMap',   VasculatureMap	...
    );

end

function Geometry = CalculateGeometry_3D( Params, SimSettings )

%==========================================================================
% Parse settings
%==========================================================================
n	=	round( SimSettings.NumMajorVessels ); % number of major vessels
N	=	Params.NumMinorVessels; % number of minor vessels

%==========================================================================
% Extract commonly used parameters for conciseness
%==========================================================================
PlotGeo     =   SimSettings.flags.PlotAnything && SimSettings.flags.PlotGeometry;
VoxelSize	=   SimSettings.VoxelSize;
SubVoxSize	=   SimSettings.SubVoxSize;
VoxelCenter	=   SimSettings.VoxelCenter;
GridSize	=   SimSettings.GridSize;
RepStr      =   num2str( Params.Rep );

%==========================================================================
% Calculate Cylinders
%==========================================================================
%   p [3 x n+N] is a point on the cylinder axis
%   v [3 x n+N] is the axis direction
%   r [1 x n+N] is the cylinder radius
t_Cylinders = tic;

[x,y]	=	regular_grid_2D(n,false);
p0      =   [VoxelSize(1) * x(:)'; VoxelSize(2) * y(:)'; zeros(1,n,'double')];
p0      =   bsxfun( @plus, VoxelCenter(:), p0 );
v0      =	repmat( double([0;0;1]), [1,n] );
r0      =	repmat( double(Params.R_Major), [1,n] );
BVF     =   Params.Total_BVF;

isInt	=   true; % allow minor vessels to intersect
orient	=   'random'; % axis-aligned cylinders

% Minor vessel radii
R_Minor     =   SimSettings.R_Minor_mu + ...
                SimSettings.R_Minor_sig * randn(1,N,'double');

if isInt
    [p,vz,r]	=	addIntersectingCylinders( VoxelSize(:), VoxelCenter(:), ...
        R_Minor, [], orient, false, p0, v0, r0, BVF );
else
    [p,vz,r]	=	nonIntersectingCylinders( VoxelSize(:), VoxelCenter(:), ...
        R_Minor, [], orient, false, p0, v0, r0, BVF );
end
[vx,vy,vz]	=	nullVectors3D( vz );
N           =   numel(r) - numel(r0);

t_Cylinders	=   toc(t_Cylinders);
display_toc_time( t_Cylinders, 'Generating Major/Minor Cylinders ' );

%==========================================================================
% Calculate venous boolean map
%==========================================================================
%   true if point is inside a cylinder, false otherwise

t_VascMap = tic;

[ p, vz, r, vx, vy, mx, VasculatureMap, Params ] = getNonIntersectingCylinderMask( ...
    p, vz, r, vx, vy, orient, isInt, Params, SimSettings );

t_VascMap	=   toc(t_VascMap);
display_toc_time( t_VascMap, 'Generating Venous Boolean Map', 1 );

% Plot cylinder map
if PlotGeo
    try
        % Empty inputs simply plots the cylinders
        nmaj   =   SimSettings.NumMajorVessels;
        h_cyl = figure;
        h_cyl = plot_cylinders_in_box( p(:,1:nmaj), vz(:,1:nmaj), r(:,1:nmaj), VoxelSize(:), VoxelCenter(:), sprintf('Vasculature Cylinders: N = %d',numel(r)), 'r', h_cyl );
        h_cyl = plot_cylinders_in_box( p(:,nmaj+1:end), vz(:,nmaj+1:end), r(:,nmaj+1:end), VoxelSize(:), VoxelCenter(:), sprintf('Vasculature Cylinders: N = %d',numel(r)), 'b', h_cyl );
        if SimSettings.flags.SaveData
            filename	=   [ 'Cylinder_Map_Rep', RepStr ];
            save_simulation_figure( filename, h_cyl, false, SimSettings );
            close(h_cyl);
        end
    catch me
        warning(me.message);
    end
end

% Plot vasculature map
if PlotGeo
    try
        h_vasc	=	plot_vasculature_map(VoxelSize,GridSize,VasculatureMap);
        
        if SimSettings.flags.SaveData
            filename	=	[ 'Vasculature_Map_Rep', RepStr ];
            save_simulation_figure( filename, h_vasc, false, SimSettings );
            close(h_vasc);
        end
    catch me
        warning(me.message);
    end
end

%==========================================================================
% Return Geometry struct
%==========================================================================

n   =   SimSettings.NumMajorVessels;
N	=   Params.NumMinorVessels;
i1	=   1:n;
i2	=   n + (1:N);

Geometry	=   struct( ...
    'MainCylinders',    struct('p',p(:,i1),'vx',vx(:,i1),'vy',vy(:,i1),'vz',vz(:,i1),'r',r(i1),'mx',{mx(:,i1)},'bx',[],'n',n),	... %TODO: add index cell arrays 'mx'
    'MinorCylinders',   struct('p',p(:,i2),'vx',vx(:,i2),'vy',vy(:,i2),'vz',vz(:,i2),'r',r(i2),'mx',{mx(:,i2)},'bx',[],'n',N),	... %TODO: add bounding box arrays 'bx'
    'VasculatureMap',   VasculatureMap,	...
    'Timings',          struct('GenerateCylinders',t_Cylinders,'GenerateVascMap',t_VascMap)	...
    );

end

function [ p, vz, r, vx, vy, mx, VasculatureMap, Params ] = getNonIntersectingCylinderMask( ...
    p, vz, r, vx, vy, isAA, isInt, Params, SimSettings )

VoxelSize	=   SimSettings.VoxelSize;
SubVoxSize	=   SimSettings.SubVoxSize;
VoxelCenter	=   SimSettings.VoxelCenter;
GridSize	=   SimSettings.GridSize;
Nmajor     	=   SimSettings.NumMajorVessels;
isUnit      =   true;
isCentered	=   true;
prec        =   'double';

% Total Blood Volume Fraction function
BVF	=   @(x) sum(x(:))/numel(x);

% Calculate Map for Major Vessels
t_VascMap0          =   tic;
idx                 =   1:Nmajor;
[p0,vz0,vx0,vy0]	=   deal( p(:,idx), vz(:,idx), vx(:,idx), vy(:,idx) );

R0                  =   Params.R_Major;
BV_Best             =   Params.Major_BloodVol * (prod(GridSize)/SimSettings.Total_Volume);
VMap_RmajorSlice	=	@(R,p0) getCylinderMask( [GridSize(1:2),1], SubVoxSize, VoxelCenter, [VoxelSize(1:2),SubVoxSize], ...
    p0, vz0, R*ones(1,Nmajor), vx0, vy0, isUnit, isCentered, prec, [] );
get_BV_RmajorSlice  =   @(R,p0) GridSize(3) * sum(reshape(VMap_RmajorSlice(R,p0),[],1));
% VMap_Rmajor       =   @(R,p0) getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
%     p0, vz0, R*ones(1,Nmajor), vx0, vy0, isUnit, isCentered, prec, [] );
% get_BV_Rmajor     =   @(R,p0) sum(reshape(VMap_Rmajor(R,p0),[],1));

BV_CurrentBest   = -inf;
p0_CurrentBest   =  p0;
R_CurrentBest    =  R0 * ones(1,Nmajor);

iters = 0;
iters_MAX = 50;
while (iters < iters_MAX) && (abs(BV_CurrentBest - BV_Best) > 0.25 * (1/max(GridSize)) * BV_Best)
    
    iters = iters + 1;
    
    if iters > 1, p1 = p0 + SubVoxSize * [2*rand(2,size(p0,2))-1; zeros(1,size(p0,2))];
    else          p1 = p0;
    end
    
    [Rlow,Rhigh]	=   deal( 0.5 * R0, 1.5 * R0 );
    %[BVlow,BVhigh]	=   deal( get_BV_Rmajor(Rlow,p1), get_BV_Rmajor(Rhigh,p1) );
    [BVlow,BVhigh]	=   deal( get_BV_RmajorSlice(Rlow,p1), get_BV_RmajorSlice(Rhigh,p1) );
    
    [BVlow_last,BVhigh_last]	=   deal(-1);
    
    while ~(BVlow == BVlow_last && BVhigh == BVhigh_last)
        [BVlow_last,BVhigh_last]	=   deal(BVlow,BVhigh);
        
        Rmid	=   (Rlow + Rhigh)/2;
        %BVmid	=   get_BV_Rmajor(Rmid,p1);
        BVmid	=   get_BV_RmajorSlice(Rmid,p1);
        
        if BVmid >= BV_Best
            BVhigh	=   BVmid;
            Rhigh	=   Rmid;
        else
            BVlow	=   BVmid;
            Rlow	=   Rmid;
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

r0             	=   R_CurrentBest * ones(1,Nmajor);
p0             	=   p0_CurrentBest;
p(:,1:Nmajor)   =   p0;
r(1,1:Nmajor)   =   r0;

Nminor          =   numel(r) - Nmajor;
VasculatureMap	=   [];
mx              =   cell(1,Nmajor+Nminor);
for ii = 1:Nmajor
    [ VasculatureMap, mx{ii} ]	=	getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
        p0(:,ii), vz0(:,ii), r0(ii), vx0(:,ii), vy0(:,ii), isUnit, isCentered, prec, VasculatureMap );
end
Major_BVF	=   BVF(VasculatureMap);

% Add minor vasculature to Map
for ii = Nmajor + (1:Nminor)
    [ VasculatureMap, mx{ii} ]  =   getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
        p(:,ii), vz(:,ii), r(ii), vx(:,ii), vy(:,ii), isUnit, isCentered, prec, VasculatureMap );
end
display_toc_time( toc(t_VascMap0), 'Venous Boolean Map: Original Cyls', 0 );

Minor_BVF	=   BVF(VasculatureMap) - Major_BVF;

% We now have an initial guess for the Minor_BVF; next we create more
% cylinders until there are at least as many as are required
t_VascMap       =   tic;
Total_BVF_Buff  =   0.05 * Params.Total_BVF * (1 - Params.Total_BVF);
NumMinorAdded   =   0;
while Minor_BVF + Major_BVF < Params.Total_BVF + Total_BVF_Buff
    R_Minor     =   SimSettings.R_Minor_mu + ...
        SimSettings.R_Minor_sig * randn(1,1,'double');
    
    if isInt
        [p1,vz1,r1] =   addIntersectingCylinders( VoxelSize, VoxelCenter, ...
            R_Minor, [], isAA, false, p0, vz0, r0 );
        [p,vz,r]    =   deal([p,p1(:,end)],[vz,vz1(:,end)],[r,r1(end)]);
    else
        [p,vz,r]    =   nonIntersectingCylinders( VoxelSize, VoxelCenter, ...
            R_Minor, [], isAA, false, p, vz, r );
    end
    [vx(:,end+1),vy(:,end+1),vz(:,end)] =   nullVectors3D( vz(:,end) );
    
    [ VasculatureMap, mx{end+1} ]   =   getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
        p(:,end), vz(:,end), r(end), vx(:,end), vy(:,end), isUnit, isCentered, prec, VasculatureMap );
    
    NumMinorAdded	=   NumMinorAdded + 1;
    Minor_BVF       =   BVF(VasculatureMap) - Major_BVF;
end
Nminor	=   numel(r) - Nmajor;

Minor_BVF_Best          =   Params.Minor_BloodVol / SimSettings.Total_Volume;
Minor_BVF_CurrentBest   =   -inf;
ind_List_CurrentBest    =   [];

Mean_Minor_BVF =   mean(cellfun(@numel,mx(Nmajor+1:end))) / prod(GridSize);
Mean_Nminor    =   floor(Minor_BVF_Best/Mean_Minor_BVF);

loopiters = 0;
loopiters_MAX = 10;
while (loopiters < loopiters_MAX) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Best) > 0.25 * (1/max(GridSize)) * Minor_BVF_Best)
    loopiters = loopiters + 1;
    MinorPerm = Nmajor + randperm(Nminor);
    
    Minor_Map0     =   false(size(VasculatureMap));
    for ind = MinorPerm(1:Mean_Nminor)
        Minor_Map0(mx{ind}) = true;
    end
    Minor_BVF0  = BVF(Minor_Map0);
    
    iters = 0;
    MinorIndsLeft = sort(MinorPerm(Mean_Nminor+1:end));
    while (iters < 10) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Best) > 0.25 * (1/max(GridSize)) * Minor_BVF_Best)
        iters = iters + 1;
        
        count  = 0;
        IndsLeft = MinorIndsLeft;
        ind_List = [];
        
        Minor_Map = Minor_Map0;
        Minor_BVF = Minor_BVF0;
        Minor_BVF_Last = -inf;
        
        while Minor_BVF < Minor_BVF_Best && ~isempty(IndsLeft)
            count = count + 1;
            Minor_BVF_Last = Minor_BVF;
            
            ix  = randi(numel(IndsLeft));
            ind = IndsLeft(ix);
            ind_List = [ind_List, ind];
            IndsLeft(ix) = [];
            
            Minor_Map(mx{ind}) = true;
            Minor_BVF = BVF(Minor_Map);
        end
        
        if abs(Minor_BVF_Last - Minor_BVF_Best) < abs(Minor_BVF - Minor_BVF_Best)
            ind_List = ind_List(1:end-1);
            Minor_BVF = Minor_BVF_Last;
        end
        
        if abs(Minor_BVF - Minor_BVF_Best) < abs(Minor_BVF_CurrentBest - Minor_BVF_Best)
            Minor_BVF_CurrentBest = Minor_BVF;
            ind_List_CurrentBest  = [sort(MinorPerm(1:Mean_Nminor)), sort(ind_List)];
        end
    end
end
if loopiters >= loopiters_MAX
    warning('Number of iterations exceeded for: Minor Vasculature');
end

Minor_BVF = Minor_BVF_CurrentBest;
Nminor = numel(ind_List_CurrentBest);
Params.NumMinorVessels = Nminor;

idx = [1:Nmajor, ind_List_CurrentBest];
[p,vz,r,mx]     =   deal(p(:,idx), vz(:,idx), r(idx), mx(idx));
[vx,vy,vz]      =   nullVectors3D( vz );
VasculatureMap  =   getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
    p, vz, r, vx, vy, isUnit, isCentered, prec, [] );

display_toc_time( toc(t_VascMap), 'Venous Boolean Map: Added Cyls   ', 0 );

[ms,me,Ms,Me,Ts,Te]	=   deal( Minor_BVF, Params.MinorVessel_RelBVF * Params.Total_BVF, ...
    Major_BVF, (1.0-Params.MinorVessel_RelBVF) * Params.Total_BVF, ...
    Minor_BVF + Major_BVF, Params.Total_BVF );

relerr	=   @(x,y) max(abs(x-y)./max(abs(x),abs(y)));

fprintf( '\n' );
fprintf( 'Minor BVF simulated:\t%0.6f\n', ms );
fprintf( 'Minor BVF expected: \t%0.6f\n', me );
fprintf( 'Minor BVF rel-error:\t%0.6f\n', relerr(ms,me) );
fprintf( 'Major BVF simulated:\t%0.6f\n', Ms );
fprintf( 'Major BVF expected: \t%0.6f\n', Me );
fprintf( 'Major BVF rel-error:\t%0.6f\n', relerr(Ms,Me) );
fprintf( 'Total BVF simulated:\t%0.6f\n', Ts );
fprintf( 'Total BVF expected: \t%0.6f\n', Te );
fprintf( 'Total BVF rel-error:\t%0.6f\n', relerr(Ts,Te) );
fprintf( '\n' );

end

function Map = VasculatureMapFunction( ...
    Params,SimSettings,x,y,z,p,vz,r,vx,vy)

Points	=	[x(:)';y(:)';z(:)'];
Map     =	isPointInCylinder( Points, p, vz, r, vx, vy, true );
Map     =	reshape( Map, size(x) );

end

function Map = SmoothVasculatureMapFunction( ...
    Params,SimSettings,x,y,z,p,vz,r,vx,vy)

persistent CALLNUM
if isempty(CALLNUM), CALLNUM = 0; end
CALLNUM = CALLNUM + 1;
fprintf('Call Number: %d\n',CALLNUM);

% Smooth inclusion function
SoftBox	=	@(x,w,s) 0.5 * erfc((x-w)/(s*sqrt(2))); % Positive side only
% SoftBox	=	@(x,w,s) 0.5 * (erf((x+w)./(sqrt(2)*s))-erf((x-w)./(sqrt(2)*s))); % Two sided
sig     =   SimSettings.R_Minor_mu * SimSettings.Smoothing;

% work in column basis for speed
[p,vx,vy,vz]	=   deal( p', vx', vy', vz' );
P	=   size(p,1);

Pts	=	[x(:),y(:),z(:)];
N	=   size(Pts,1);
Map	=	zeros(N,1);

for ii = 1:P
    % Get rejection (shortest vectors from line L(t)=p+vz*t to Points)
    V_T	=   [vx(ii,:); vy(ii,:); vz(ii,:)]';
    Pts	=	bsxfun( @minus, Pts, p(ii,:) );
    Pts	=   Pts * V_T;
    
    % Transform cylinder axes/points as well
    p	=   bsxfun( @minus, p, p(ii,:) );
    [p,vx,vy,vz]	=   deal( p*V_T, vx*V_T, vy*V_T, vz*V_T );
    
    % If rejection is shorter than cylinder radius, point is in cylinder
    R	=   hypot( Pts(:,1), Pts(:,2) );
    Map	=	Map + SoftBox(R,r(ii),sig);
end
Map	=   reshape(Map,size(x));

end

function h = plot_vasculature_map(VoxelSize,GridSize,VasculatureMap,AllowSkip)

if nargin < 4; AllowSkip = true; end

[X,Y,Z]	=	meshgrid( linspace(0,VoxelSize(1),GridSize(1)), ...
                      linspace(0,VoxelSize(2),GridSize(2)), ...
                      linspace(0,VoxelSize(3),GridSize(3)) );
X	=	X(VasculatureMap);
Y	=	Y(VasculatureMap);
Z	=	Z(VasculatureMap);

if AllowSkip
    MaxPointsToPlot	=   1e6;
    if numel(X) > MaxPointsToPlot
        skip	=	ceil( numel(X)/MaxPointsToPlot );
        [X,Y,Z]	=	deal(X(1:skip:end),Y(1:skip:end),Z(1:skip:end));
    end
end

h	=	figure;
plot3(X,Y,Z,'b.','markersize',10); axis image; drawnow
title('Discrete Map of Vasculature');

end

function [isLoadGeometry,Geometry] = checkLoadGeometry( SimSettings )

isMinimization	=	~any(isnan( SimSettings.MinimizationType ));
iteration       =   SimSettings.MinimizationIter;

isLoadGeometry	=	false;
Geometry        =   [];
if ~isMinimization || (isMinimization && iteration == 1)
    return
end

lastsavepath	=   [ SimSettings.RootPath, '/', ...
        SimSettings.MinimizationType, '_iter_', num2strpad(iteration-1,3) ];

ThisParams      =   table2array( SimSettings.InitialParams );
LastSettings	=   load( [ lastsavepath, '/SimSettings' ] );
LastParams      =   table2array( LastSettings.SimSettings.InitialParams );

lb	=	SimSettings.MinimizationOpts.LowerBound;
ub	=   SimSettings.MinimizationOpts.UpperBound;
tol	=   SimSettings.MinimizationOpts.options.FinDiffRelStep;

x	=   (ThisParams - lb)./(ub-lb);
y	=   (LastParams - lb)./(ub-lb);
e	=   max(abs(x-y)./max(abs(x),abs(y)));

if e < 10 * tol
    % Evaluation of gradient; re-load last Geometry
    isLoadGeometry	=   true;
    g               =   load([lastsavepath,'/ParamSet_1_1/Geometry_Rep1.mat']);
    Geometry        =   g.Geometry;
    clear g
    
    M               =   Geometry.MainCylinders;
    m               =   Geometry.MinorCylinders;
    [p,vz,r,vx,vy]  =	deal( [M.p,m.p], [M.vz,m.vz], [M.r,m.r], [M.vx,m.vx], [M.vy,m.vy] );
    
    VoxelSize	=   SimSettings.VoxelSize;
    SubVoxSize	=   SimSettings.SubVoxSize;
    VoxelCenter	=   SimSettings.VoxelCenter;
    GridSize	=   SimSettings.GridSize;
    isUnit      =   true;
    isCentered	=   true;
    prec        =   'double';
    Geometry.VasculatureMap	=	...
        getCylinderMask( GridSize, SubVoxSize, VoxelCenter, VoxelSize, ...
        p, vz, r, vx, vy, isUnit, isCentered, prec, [] );
end

end
