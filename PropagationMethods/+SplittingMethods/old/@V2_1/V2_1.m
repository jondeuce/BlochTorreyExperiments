classdef V2_1
    %V2_1 Second order operator splitting method for the BT equation
    %consisting of two multiplication stages and one convolution stage
    
    properties (GetAccess = public, SetAccess = private)
        Dcoeff
        TimeStep
        NSubSteps
        GridSize
        VoxelSize
    end
    
    properties (GetAccess = public, SetAccess = private)
        E1
        K1
    end
    
    properties (GetAccess = private, SetAccess = immutable)
        a1
        b1
        allowPreCompKernels
        allowPreCompDecays
    end
    
    methods
        
        function V = V2_1(h, n, D, Gsize, Vsize, Gamma, pckernels, pcdecays, allowpckernels, allowpcdecays)
            
            if nargin < 10; allowpcdecays = false; end
            if nargin < 9; allowpckernels = false; end
            if nargin < 8; pcdecays = false; end
            if nargin < 7; pckernels = false; end
            
            if pckernels && ~allowpckernels; warning('Not pre-computing kernels: allowPreCompKernels == false'); end
            if pcdecays  && ~allowpcdecays;  warning('Not pre-computing decays: allowPreCompKernels == false'); end
                        
            V.b1 = 0.5;
            V.a1 = 1.0;
            
            V.TimeStep  = h;
            V.NSubSteps = n;
            V.Dcoeff    = D;
            V.GridSize  = Gsize;
            V.VoxelSize = Vsize;
            
            V.allowPreCompKernels = allowpckernels;
            V.allowPreCompDecays = allowpcdecays;
            
            % ---- Precompute ---- %
            if pckernels && V.allowPreCompKernels
                V = precomputeKernels(V,h,D);
            else
                V = computeKernels(V,h,D);
            end
            
            if pcdecays && V.allowPreCompDecays
                V = precomputeDecays(V,h,Gamma);
            end
            
        end
        
        function V = computeKernels(V, h, D)
            
            V.TimeStep = h;
            V.Dcoeff   = D;
            V.K1 = Geometry.GaussianKernel(sqrt(2*(V.a1*D)*h),V.GridSize,V.VoxelSize);
            
        end
        
        function V = precomputeKernels(V, h, D)
            
            V    = computeKernels(V, h, D);
            V.K1 = precompute(V.K1); %GaussianKernel class handles precomputation
            
        end
        
        function V = precomputeDecays(V, h, Gamma)
            
            V.TimeStep = h;
            V.E1 = exp( (-V.b1 * h) .* Gamma );
            
        end
                
    end
    
end

