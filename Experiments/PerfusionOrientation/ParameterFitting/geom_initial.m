function [ Geom, GeomArgs, x0, lb, ub ] = geom_initial( GeomArgs, OptVariables, x0, lb, ub, varargin )
%GEOM_INITIAL [ Geom, GeomArgs, x0, lb, ub ] = geom_initial( GeomArgs, OptVariables, x0, lb, ub )

for ii = 1:2:length(varargin)
    GeomArgs.(varargin{ii}) = varargin{ii+1};    
end

% Generate initial geometry and transformed optimization bounds
switch upper(OptVariables)
    case 'CA_IBVF_ABVF'
        % Initial geometry is simply the supplied arguments; to
        % transformation of bounds is needed
        GeomArgs.iBVF = x0(2);
        GeomArgs.aBVF = x0(3);
        GeomNameValueArgs = struct2arglist(GeomArgs);
        Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );
        
    case 'CA_RMAJOR_MINOREXPANSION'
        GeomArgs.ImproveMajorBVF = true;
        GeomArgs.ImproveMinorBVF = true;
        
        % Generate geometry for expansion
        GeomArgs.iBVF = ub(2);
        GeomArgs.aBVF = ub(3);
        GeomNameValueArgs = struct2arglist(GeomArgs);
        Geom = Geometry.CylindricalVesselFilledVoxel( GeomNameValueArgs{:} );
        
        % Generate new initial guesses and bounds via empirical models
        getRmajor0 = @(aBVF) sqrt( prod(VoxelSize) * aBVF / ( Nmajor * pi * rayBoxIntersectionLength( VoxelCenter(:), [sind(MajorAngle); 0; cosd(MajorAngle)], VoxelSize(:), VoxelCenter(:) ) ) );
        getSpaceFactor0 = @(iBVF) (Geom.iBVF/iBVF)^(1/2.3); % empirical model: iBVF = iBVF_max * SpaceFactor^(-2.3)
        
        lb_old = lb;
        x0_old = x0;
        ub_old = ub;
        lb = [lb_old(1), getRmajor0(lb_old(3)), getSpaceFactor0(ub_old(2))];
        x0 = [x0_old(1), getRmajor0(aBVF0),     getSpaceFactor0(iBVF0)];
        ub = [ub_old(1), getRmajor0(ub_old(3)), getSpaceFactor0(lb_old(2))];
        
    otherwise
        error('''OptVariables'' must be ''CA_iBVF_aBVF'' or ''CA_Rmajor_MinorExpansion''');
end

end
