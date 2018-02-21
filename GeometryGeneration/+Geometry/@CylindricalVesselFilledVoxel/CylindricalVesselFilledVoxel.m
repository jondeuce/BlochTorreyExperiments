classdef CylindricalVesselFilledVoxel
    %CYLINDRICALVESSELFILLEDVOXEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties ( GetAccess = public, SetAccess = private )
        GridSize          % Grid size, e.g. [512,512,512]
        VoxelSize         % Voxel dimensions (unitful), e.g. [3000,3000,3000] um
        VoxelCenter       % Center of Voxel, e.g. [1500, 1500, 1500] um
        SubVoxSize        % Size of subvoxels, e.g. [3000/512,3000/512,3000/512] um
        Nmajor            % Number of major vessels
        NumMajorArteries  % Number of major vessels which are arteries
        NumMinorArteries  % Number of minor vessels which are arteries
        MinorArterialFrac % (Target) Fraction of minor vessels which are arteries
        seed              % Random number generator seed used in constructor
    end
    
    properties ( GetAccess = public, SetAccess = private )
        P, R, Vx, Vy, Vz, Idx  % All Cylinder parameters
        N, Nminor % Total number of cylinders/Number of minor cylinders
        BVF, iBVF, aBVF % Total BVF, isotropic BVF, and anisotropic BVF
        iRBVF, aRBVF% relative isotropic/anisotropic BVF fractions
        Rmajor % Major vessel radii (scalar)
        Rminor_mu, Rminor_sig % mean/std of minor cylinder radii
        RmajorFun, RminorFun % Functions for generating major/minor radii
        MinorDilation, MinorRadiusFactor, isMinorDilated % Minor dilation settings
        MajorArteries, MinorArteries % Lists of vessels which are arteries
        ArterialIndices % Indices for all arteries
        VasculatureMap % Boolean map which is 1 at points containings vessels, and 0 elsewhere
        MetaData % Field for storing arbitrary MetaData of interest
    end
    
    properties ( GetAccess = public, SetAccess = private )
        InitGuesses % Initial guesses for generating major/minor cyls
        Targets % Target BVF, etc. for vasculature map
        opts % Keyword arguments
    end
    
    properties ( Dependent = true, GetAccess = public, SetAccess = private )
        p,  r,  vx,  vy,  vz,  idx  % Minor Cylinder parameters
        p0, r0, vx0, vy0, vz0, idx0 % Major Cylinder parameters
    end
    
    properties ( GetAccess = private, SetAccess = immutable, Hidden = true )
        SimSettings % SimSettings that was passed to constructor
        Params      % Params that was passed to constructor
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function [ G ] = CylindricalVesselFilledVoxel( varargin )
            % CylindricalVesselFilledVoxel class constructor.
            
            %==============================================================
            % Parse keyword arguments and get options structure
            %==============================================================
            G = parseinputs(G, varargin{:});
                        
            %==============================================================
            % Fill geometry with random cylinders
            %==============================================================
            G = FillWithRandomCylinders(G);
            
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SETTERS/GETTERS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        % Overloads for major cylinders
        function p0 = get.p0(G)
            p0 = G.P(:,1:G.Nmajor);
        end
        function r0 = get.r0(G)
            r0 = G.R(:,1:G.Nmajor);
        end
        function vx0 = get.vx0(G)
            vx0 = G.Vx(:,1:G.Nmajor);
        end
        function vy0 = get.vy0(G)
            vy0 = G.Vy(:,1:G.Nmajor);
        end
        function vz0 = get.vz0(G)
            vz0 = G.Vz(:,1:G.Nmajor);
        end
        function idx0 = get.idx0(G)
            idx0 = G.Idx(:,1:G.Nmajor);
        end
        
        function G = set.p0(G,in)
            G.P(:,1:G.Nmajor) = in;
        end
        function G = set.r0(G,in)
            G.R(:,1:G.Nmajor) = in;
        end
        function G = set.vx0(G,in)
            G.Vx(:,1:G.Nmajor) = in;
        end
        function G = set.vy0(G,in)
            G.Vy(:,1:G.Nmajor) = in;
        end
        function G = set.vz0(G,in)
            G.Vz(:,1:G.Nmajor) = in;
        end
        function G = set.idx0(G,in)
            G.Idx(:,1:G.Nmajor) = in;
        end
        
        % Overloads for minor cylinders
        function p = get.p(G)
            p = G.P(:,G.Nmajor+1:end);
        end
        function r = get.r(G)
            r = G.R(:,G.Nmajor+1:end);
        end
        function vx = get.vx(G)
            vx = G.Vx(:,G.Nmajor+1:end);
        end
        function vy = get.vy(G)
            vy = G.Vy(:,G.Nmajor+1:end);
        end
        function vz = get.vz(G)
            vz = G.Vz(:,G.Nmajor+1:end);
        end
        function idx = get.idx(G)
            idx = G.Idx(:,G.Nmajor+1:end);
        end
        
        function G = set.p(G,in)
            G.P(:,G.Nmajor+1:end) = in;
        end
        function G = set.r(G,in)
            G.R(:,G.Nmajor+1:end) = in;
        end
        function G = set.vx(G,in)
            G.Vx(:,G.Nmajor+1:end) = in;
        end
        function G = set.vy(G,in)
            G.Vy(:,G.Nmajor+1:end) = in;
        end
        function G = set.vz(G,in)
            G.Vz(:,G.Nmajor+1:end) = in;
        end
        function G = set.idx(G,in)
            G.Idx(:,G.Nmajor+1:end) = in;
        end
    end
    
    
end

