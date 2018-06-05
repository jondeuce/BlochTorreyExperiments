classdef ExpmvStepper
    %EXPMVSTEPPER Exponential matrix stepping method.
    
    properties (GetAccess = public, SetAccess = immutable)
        opts % Parsed input options
    end
    
    properties (GetAccess = public, SetAccess = private)
        A         % Matrix or object representing the action of a matrix
        TimeStep  % Time step
        GridSize  % Underlying grid size (may be empty)
        VoxelSize % Underlying voxel dimensions (may be empty)
    end
    
    properties (GetAccess = private, SetAccess = private, Hidden = true)
        isprecomputed  % Flag indicating if M was been precomputed
        selectdegargs  % Parsed arguments for select_taylor_degree
        expmvargs      % Parsed arguments for expmv
    end
    
    methods
        
        % ----- Constructor ----- %
        function V = ExpmvStepper(t, A, gsize, vdims, varargin)
            % V = ExpmvStepper(t, A, gsize, vsize, varargin)
            % REQUIRED POSITIONAL ARGS:
            % 	A       Matrix or object representing the action of a matrix
            %   t       Time step
            % 	gsize   Underlying grid size (may be empty)
            %   vdims   Underlying voxel dimensions (may be empty)
            %
            % OPTIONAL KEYWORD ARGS:
            %   TODO
            
            [V.opts, V.selectdegargs, V.expmvargs] = bt_expmv_parseinputs(varargin{:});
            
            V.A = A;
            V.TimeStep = t;
            V.GridSize = gsize;
            V.VoxelSize = vdims;
            V.isprecomputed = false;
            
        end
                
        function V = precompute(V,b)
            
            V.expmvargs{1}  = select_taylor_degree(V.A,b,V.selectdegargs{:});
            V.isprecomputed = true;
            
        end
                
        function V = clearPrecomputation(V)
            V.expmvargs{1}  = [];
            V.isprecomputed = false;
        end
                
    end
        
    methods (Static = true)
        % ----- Testing ----- %
        test
    end
    
end

