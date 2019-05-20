classdef BTSplitStepper
    %BTSPLITSTEPPER N-th order symmetric operator splitting method for the
    %Bloch-Torrey equation consisting of 2N multiplication stages and
    %2N-1 convolution stages, for a total of 4N-1 operations per step.
    
    properties (GetAccess = public, SetAccess = immutable)
        TimeStep
        Order
        GridSize
        VoxelSize
        NReps
        a % Convolution step coefficients (vector of length M)
        b % Exponential step coefficients (vector of length N)
        allowPreCompConvKernels
        allowPreCompExpArrays
        useGaussianKernels
    end
    
    properties (GetAccess = public, SetAccess = private)
        ExpArrays   % Exponentials for the 2N multiplication substeps
        ExpChain    % Exponential for chaining together substeps (if NReps > 1)
        ConvKernels % Precomputed GaussianKernel objects
        GammaDerivs % Precomputed and scaled gamma derivatives
    end
    
    methods
        
        % ----- Constructor ----- %
        function BTStepper = BTSplitStepper(varargin)
            % BTStepper = BTSplitStepper(dt, Dcoeff, Gamma, dGamma, GridSize, VoxelSize, varargin)
            % REQUIRED POSITIONAL ARGS:
            %   dt:         Time step
            %   Dcoeff:     Diffusion Coefficient [um^2/s]
            %   Gamma:      Complex decay [rad/s]
            %   dGamma:     Complex decay derivative [(rad/s)/[param units]]
            %   GridSize:   Size of the grid (in voxels)
            %   VoxelSize:  Size of the voxel (in um)
            %
            % OPTIONAL KEYWORD ARGS:
            %   Order [default = 2] Order of stepper
            % 	NReps [default = 1] Number of repetitions of step of size 'dt'
            %   allowPreCompConvKernels [default = true]
            %   allowPreCompExpArrays [default = 2]
            
            args = parseinputs(BTStepper, varargin{:});
            BTfields = fieldnames(BTStepper);
            for f = fieldnames(args).'
                fname = f{1};
                % 'isfield' does not work for user defined classes
                if ismember(fname,BTfields)
                    BTStepper.(fname) = args.(fname);
                end
            end
            
            % ---- Precompute ---- %
            BTStepper = precomputeConvKernels(BTStepper,args.Dcoeff);
            BTStepper = precomputeExpDecays(BTStepper,args.Gamma);
            BTStepper = precomputeGammaDerivs(BTStepper,args.dGamma);
            
        end
        
        % ----- Kernel Methods ----- %
        function V = computeKernels(V,Dcoeff)
            % V = computeKernels(V,Dcoeff)
            
            % GaussianKernel objects handle their own precomputation, so we
            % can create lightweight un-precomputed objects first which
            % will function with or without precomputation
            
            if ~isempty(Dcoeff)
                M = length(V.a);
                V.ConvKernels = cell(M,1);
                for n = 1:M
                    dt = V.a(n) * V.TimeStep;
                    if V.useGaussianKernels
                        % Create GaussianKernel object
                        sigma = sqrt(2 * Dcoeff * dt); % Standard deviation of kernel (unitful, possibly complex)
                        V.ConvKernels{n} = Geometry.GaussianKernel(sigma, V.GridSize, V.VoxelSize);
                    else
                        % Create KroneckerPropagator object
                        V.ConvKernels{n} = Geometry.KroneckerPropagator.laplacianPropagator(Dcoeff, dt, V.GridSize, V.VoxelSize);
                    end
                end
            end
            
        end
        
        function V = precomputeConvKernels(V,Dcoeff)
            % V = precomputeConvKernels(V,Dcoeff)
            
            V = computeKernels(V,Dcoeff);
            
            if V.allowPreCompConvKernels && ~isempty(Dcoeff)
                M = length(V.a);
                for m = 1:M
                    V.ConvKernels{m} = precompute(V.ConvKernels{m});
                end
            end
            
        end
        
        function V = precomputeExpDecays(V,Gamma)
            % V = precomputeExpDecays(V,Gamma)
            
            if V.allowPreCompExpArrays && ~isempty(Gamma)
                N = length(V.b);
                V.ExpArrays = cell(N,1);
                for n = 1:N
                    V.ExpArrays{n} = exp( (-V.b(n) * V.TimeStep) .* Gamma );
                end
                
                if V.NReps > 1
                    V.ExpChain = exp( (-2*V.b(1) * V.TimeStep) .* Gamma );
                end
            end
            
        end
        
        function V = precomputeGammaDerivs(V,dGamma)
            % V = precomputeGammaDerivs(V,dGamma)
            
            if ~isempty(dGamma)
                if ~isa(dGamma,'cell')
                    V.GammaDerivs = { (-V.TimeStep/2) * dGamma };
                else
                    V.GammaDerivs = cell(numel(dGamma),1);
                    for ii = 1:numel(dGamma)
                        V.GammaDerivs{ii} = (-V.TimeStep/2) * dGamma{ii};
                    end
                end
            else
                V.GammaDerivs = {};
            end
            
        end
        
        function V = clearExpArrays(V)
            V.ExpArrays = {};
        end
        
        function V = clearExpChain(V)
            V.ExpChain = [];
        end
        
        function V = clearConvKernels(V)
            V.ConvKernels = {};
        end
        
        function V = clearGammaDerivs(V)
            V.GammaDerivs = {};
        end
        
    end
    
    methods (Access = private)
        % ----- Parsing keyword args ----- %
        args = parseinputs(BTStepper, varargin)
    end
    
    methods (Static = true)
        % ----- Testing ----- %
        test()
    end
    
end

