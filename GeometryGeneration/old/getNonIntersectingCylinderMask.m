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
