function [ G ] = ImproveMinorBVF( G )
%IMPROVEMINORBVF 

% Ensure cylinder direction vectors are unit, and set parameters
G           =   NormalizeCylinderVecs(G);
isUnit      =   true;
isCentered  =   true;
prec        =   'double';

% Currently only have implemented the case where minor vessels can
% self-intersect; the amount of self-intersection, however, is negligeable
% in general
if ~G.opts.AllowMinorSelfIntersect
    warning('Minor vessels will NOT be allowed to intersect with themselves');
end

% Total Blood Volume Fraction function
vec     = @(x) x(:);
BVF_Fun = @(x) sum(vec(sum(x,1)))/numel(x); % faster than sum(x(:)) for logical arrays

% Calculate initial vasc. map and blood volumes
[ G ] = CalculateVasculatureMap( G );

% We now have an initial guess for the G.iBVF; next we create more
% cylinders until there are at least as many as are required
BVF_RelErr = 0.05 * (1.0 - G.Targets.BVF); % 5 percent relative to gap between G.Targets.BVF and 1.0
while G.iBVF + G.aBVF < G.Targets.BVF * ( 1.0 + BVF_RelErr )
    R_Minor = G.RminorFun(1,1,'double');
    
    if G.opts.AllowMinorSelfIntersect
        if G.opts.AllowMinorMajorIntersect
            [P,Vz,R] = addIntersectingCylinders( G.VoxelSize, G.VoxelCenter, ...
                R_Minor, [], 'random');
        else
            [P,Vz,R] = addIntersectingCylinders( G.VoxelSize, G.VoxelCenter, ...
                R_Minor, [], 'random', false, G.p0, G.vz0, G.r0 );
        end
        [G.P,G.Vz,G.R] = deal([G.P,P(:,end)],[G.Vz,Vz(:,end)],[G.R,R(end)]);
    else
        [G.P,G.Vz,G.R] = nonIntersectingCylinders( G.VoxelSize, G.VoxelCenter, ...
            R_Minor, [], [], false, G.P, G.Vz, G.R );
    end
    [G.Vx(:,end+1),G.Vy(:,end+1),G.Vz(:,end)] = nullVectors3D( G.Vz(:,end) );
    
    [ G.VasculatureMap, G.Idx ]   =   getCylinderMask( G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
        G.P(:,end), G.Vz(:,end), G.R(end), G.Vx(:,end), G.Vy(:,end), isUnit, isCentered, prec, G.VasculatureMap, G.Idx );
    
    G.iBVF       =   BVF_Fun(G.VasculatureMap) - G.aBVF;
end
G.N = numel(G.R);
G.Nminor    =   G.N - G.Nmajor;

Minor_BVF_Best          =   G.Targets.iBVF;
Minor_BVF_CurrentBest   =   -inf;
ind_List_CurrentBest    =   [];

Mean_Minor_BVF =   mean(cellfun(@numel,G.Idx(G.Nmajor+1:end))) / prod(G.GridSize);
Mean_Nminor    =   floor(Minor_BVF_Best/Mean_Minor_BVF);

% Remove and re-insert new minor vessels to further improve BVF
loopiters = 0;
loopiters_MAX = 10;
Minor_BVF_ErrTol = 1e-8 * (G.SubVoxSize/max(G.GridSize)) * Minor_BVF_Best;
while (loopiters < loopiters_MAX) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Best) > Minor_BVF_ErrTol)
    loopiters = loopiters + 1;
    MinorPerm = G.Nmajor + randperm(G.Nminor);
    
    Minor_Map0     =   false(G.GridSize);
    for ind = MinorPerm(1:Mean_Nminor)
        Minor_Map0(G.Idx{ind}) = true;
    end
    Minor_BVF0  = BVF_Fun(Minor_Map0);
    
    iters = 0;
    MinorIndsLeft = sort(MinorPerm(Mean_Nminor+1:end));
    while (iters < 10) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Best) > Minor_BVF_ErrTol)
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
            
            Minor_Map(G.Idx{ind}) = true;
            Minor_BVF = BVF_Fun(Minor_Map);
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
    % These iterations are fast, so the tolerance is set exceedingly low on
    % purpose; expect to reach the MAX iters every time; no need to warn
    
    % warning('Number of iterations exceeded for: Minor Vasculature');
end

G.iBVF = Minor_BVF_CurrentBest;
G.Nminor = numel(ind_List_CurrentBest);
G.N = G.Nmajor + G.Nminor;

idx = [1:G.Nmajor, ind_List_CurrentBest];
[G.P,G.Vz,G.R] = deal(G.P(:,idx), G.Vz(:,idx), G.R(idx));
G = NormalizeCylinderVecs(G);

G.BVF   = G.iBVF + G.aBVF;
G.iRBVF = G.iBVF / G.BVF;
G.aRBVF = G.aBVF / G.BVF;

end
