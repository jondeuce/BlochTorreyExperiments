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

% We now have an initial guess; next we create more cylinders until there
% are at least as many as are required to reach the target iBVF
BVF_RelErr = 0.05 * (1.0 - G.Targets.BVF); % 5 percent relative to gap between G.Targets.BVF and 1.0
while G.iBVF < G.Targets.iBVF * ( 1.0 + BVF_RelErr )
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
    
    G.BVF  = BVF_Fun(G.VasculatureMap);
    G.iBVF = G.BVF - G.aBVF;
end
G.N = numel(G.R);
G.Nminor = G.N - G.Nmajor;

% ======================================================================= %
% Old method of randomly adding/removing cylinders
% ======================================================================= %

% Minor_BVF_Goal          =   G.Targets.iBVF;
% Minor_BVF_CurrentBest   =   -inf;
% ind_List_CurrentBest    =   [];
% 
% % Average BVF per minor cylinders currently, and given that, the approximate
% % number of minor cylinders to reach the desired BVF
% Mean_Minor_BVF =   (G.BVF - G.aBVF)/G.Nminor;
% Mean_Nminor    =   floor(Minor_BVF_Goal/Mean_Minor_BVF);
% 
% % Remove and re-insert new minor vessels to further improve BVF
% loopiters = 0;
% loopiters_MAX = 10;
% Minor_BVF_ErrTol = 1e-8 * (G.SubVoxSize/max(G.GridSize)) * Minor_BVF_Goal;
% getCylMask = @(IndSet) getCylinderMask( G.GridSize, G.SubVoxSize, G.VoxelCenter, G.VoxelSize, ...
%         G.P(:,IndSet), G.Vz(:,IndSet), G.R(IndSet), G.Vx(:,IndSet), G.Vy(:,IndSet), ...
%         isUnit, isCentered, prec );
% 
% while (loopiters < loopiters_MAX) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Goal) > Minor_BVF_ErrTol)
%     loopiters = loopiters + 1;
%     
% %     MinorPerm = G.Nmajor + randperm(G.Nminor);
%     MinorPerm = randperm(G.Nminor);
%     
% %     Minor_Map0     =   false(G.GridSize);
% %     for ind = MinorPerm(1:Mean_Nminor)
% %         Minor_Map0(G.idx{ind}) = true;
% %     end
% %     Minor_BVF0  = BVF_Fun(Minor_Map0);
%     
%     Vasc_Map0 = getCylMask([1:G.Nmajor, G.Nmajor + MinorPerm(1:Mean_Nminor)]);
%     Vasc_BVF0 = BVF_Fun(Vasc_Map0);
%     
%     iters = 0;
%     MinorIndsLeft = sort(MinorPerm(Mean_Nminor+1:end));
%     while (iters < 10) && (abs(Minor_BVF_CurrentBest - Minor_BVF_Goal) > Minor_BVF_ErrTol)
%         iters = iters + 1;
%         
%         IndsLeft = MinorIndsLeft;
%         ind_List = [];
%         
% %         Minor_Map = Minor_Map0;
% %         Minor_BVF = Minor_BVF0;
% %         Minor_BVF_Last = -inf;
%         
%         Vasc_Map = Vasc_Map0;
%         Vasc_BVF = Vasc_BVF0;
%         Minor_BVF = Vasc_BVF - G.aBVF;
%         Minor_BVF_Last = -inf;
%         
%         while Minor_BVF < Minor_BVF_Goal && ~isempty(IndsLeft)
%             Minor_BVF_Last = Minor_BVF;
%             
%             ix  = randi(numel(IndsLeft));
%             ind = IndsLeft(ix);
%             ind_List = [ind_List, ind];
%             IndsLeft(ix) = [];
%             
% %             Minor_Map(G.idx{ind}) = true;
% %             Minor_BVF = BVF_Fun(Minor_Map);
%             
%             Vasc_Map(G.idx{ind}) = true;
%             Vasc_BVF = BVF_Fun(Vasc_Map);
%             Minor_BVF = Vasc_BVF - G.aBVF;
%         end
%         
%         % Check which of last two iters were best
%         if abs(Minor_BVF_Last - Minor_BVF_Goal) < abs(Minor_BVF - Minor_BVF_Goal)
%             ind_List = ind_List(1:end-1);
%             Minor_BVF = Minor_BVF_Last;
%         end
%         
%         % update current best
%         if abs(Minor_BVF - Minor_BVF_Goal) < abs(Minor_BVF_CurrentBest - Minor_BVF_Goal)
%             Minor_BVF_CurrentBest = Minor_BVF;
%             ind_List_CurrentBest  = [sort(MinorPerm(1:Mean_Nminor)), sort(ind_List)];
%         end
%     end
% end
% if loopiters >= loopiters_MAX
%     % These iterations are fast, so the tolerance is set exceedingly low on
%     % purpose; expect to reach the MAX iters every time; no need to warn
%     
%     % warning('Number of iterations exceeded for: Minor Vasculature');
% end
% 
% G.iBVF = Minor_BVF_CurrentBest;
% G.Nminor = numel(ind_List_CurrentBest);
% G.N = G.Nmajor + G.Nminor;
% 
% CylIdx = [1:G.Nmajor, G.Nmajor + ind_List_CurrentBest];
% [G.P,G.Vz,G.R] = deal(G.P(:,CylIdx), G.Vz(:,CylIdx), G.R(CylIdx));
% G = NormalizeCylinderVecs(G);

% ======================================================================= %
% New method of uniformly expanding isotropic cylinders
% ======================================================================= %
VolumeFactor = G.Targets.iBVF/G.iBVF;
TolX = 1e-6 * G.SubVoxSize; % Approximate grid resolution
TolF = 1e-4 * G.Targets.iBVF; % Approximate function tolerance
G = ExpandMinorVesselsByVolumeFactor(G, VolumeFactor, TolX, TolF);
G = NormalizeCylinderVecs(G); % probably unnecessary, but it's cheap

G.BVF   = G.iBVF + G.aBVF;
G.iRBVF = G.iBVF / G.BVF;
G.aRBVF = G.aBVF / G.BVF;

end
