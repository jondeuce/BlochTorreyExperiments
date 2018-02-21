classdef V4_3
    %V4_3 Fourth order operator splitting method for the BT equation
    %consisting of four multiplication stages and three convolution stages
    
    properties (GetAccess = public, SetAccess = private)
        Dcoeff
        TimeStep
        NSubSteps
        GridSize
        VoxelSize
    end
    
    properties (GetAccess = public, SetAccess = private)
        E1, E2
        K1, K2
    end
    
    properties (GetAccess = private, SetAccess = immutable)
        a1, a2
        b1, b2
        allowPreCompKernels
        allowPreCompDecays
    end
    
    methods
        
        function V = V4_3(h, n, D, Gsize, Vsize, Gamma, pckernels, pcdecays, allowpckernels, allowpcdecays)
            
            if nargin < 10; allowpcdecays = false; end
            if nargin < 9; allowpckernels = false; end
            if nargin < 8; pcdecays = false; end
            if nargin < 7; pckernels = false; end
            
            if pckernels && ~allowpckernels; warning('Not pre-computing kernels: allowPreCompKernels == false'); end
            if pcdecays  && ~allowpcdecays;  warning('Not pre-computing decays: allowPreCompKernels == false'); end
            
            k     =  2^(1/3) * exp(2i*pi/3);
            alpha =  1/(2-k);
            beta  = -k/(2-k);
            
            V.b1 = alpha * 1/2;
            V.a1 = alpha;
            V.b2 = alpha * 1/2 + beta * 1/2;
            V.a2 = beta;
            
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
            V.K2 = Geometry.GaussianKernel(sqrt(2*(V.a2*D)*h),V.GridSize,V.VoxelSize);
            
        end
                
        function V = precomputeKernels(V, h, D)
            
            V    = computeKernels(V, h, D);
            V.K1 = precompute(V.K1);
            V.K2 = precompute(V.K2);
            
        end
        
        function V = precomputeDecays(V, h, Gamma)
            
            V.TimeStep = h;
            
            V.E1 = exp( (-V.b1 * h) .* Gamma );
            V.E2 = exp( (-V.b2 * h) .* Gamma );
            
        end
                
    end
    
end

