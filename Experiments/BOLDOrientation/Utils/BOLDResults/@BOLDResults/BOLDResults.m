classdef BOLDResults
    %BOLDRESULTS Class for storing results of a BOLDOrientation simulation.
    
    properties ( GetAccess = public, SetAccess = immutable )
        EchoTimes               % EchoTimes simulated (row vector)
        Alphas                  % Angles with the main magnetic field B0 simulated (row vector) [rads]
        DeoxyBloodLevels        % Deoxygenated blood levels simulated (row vector)
        OxyBloodLevels          % Oxygenated blood levels simulated (row vector)
        Hcts                    % Hematocrit levels simulated (row vector)
        Reps                    % Repitition index of random geometries simulated (row vector)
    end
    
    properties ( GetAccess = public, SetAccess = public )
        MetaData % Field for convieniently saving misc. metadata required to reproduce
                 % BOLDResults object, e.g. SimSettings, Params, Geometry, etc.
    end
    
    properties ( GetAccess = public, SetAccess = private )
        DeoxySignal      % Deoxygenated Signal vs. Time curves, indexed by immutable properties
        DeoxySignalIntra % Intravascular Deoxygenated Signal vs. Time curves, indexed by immutable properties
        DeoxySignalExtra % Extravascular Deoxygenated Signal vs. Time curves, indexed by immutable properties
        DeoxySignalVRS   % VRS Deoxygenated Signal vs. Time curves, indexed by immutable properties
        OxySignal        % Oxygenated Signal vs. Time curves, indexed by immutable properties
        OxySignalIntra   % Intravascular Oxygenated Signal vs. Time curves, indexed by immutable properties 
        OxySignalExtra   % Extravascular Oxygenated Signal vs. Time curves, indexed by immutable properties 
        OxySignalVRS     % VRS Oxygenated Signal vs. Time curves, indexed by immutable properties
    end
    
    properties ( GetAccess = private, SetAccess = immutable )
        NArgs % Total number of arguments (e.g. immutable public properties above)
        NTot % Total number of results stored, determined by immutable properties
        Size % Size of results stored in a cartesian grid, determined by immutable properties
        NRep % Total number of results stored per repetition
        SizeRep % Size of results stored in a cartesian grid (without repetitions)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CLASS CONSTRUCTOR:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        function R = BOLDResults( EchoTimes, Alphas, DeoxyBloodLevels, OxyBloodLevels, Hcts, Reps )
            % BOLDResults class constructor.
            
            if ~all(iswholenumber(Reps))
                error('Reps must be a whole number, or an array of whole numbers');
            end
            
            if any(Alphas(:) > pi/2)
                warning('BOLDResults excepts angles in radians, but some of the input angles exceed pi/2!');
            end
            
            R.EchoTimes = EchoTimes(:).';
            R.Alphas = Alphas(:).';
            R.DeoxyBloodLevels = DeoxyBloodLevels(:).';
            R.OxyBloodLevels = OxyBloodLevels(:).';
            R.Hcts = Hcts(:).';
            R.Reps = Reps(:).';
            
            R.NArgs = 6;
            R.Size = [numel(EchoTimes), numel(Alphas), numel(DeoxyBloodLevels), numel(OxyBloodLevels), numel(Hcts), numel(Reps) ];
            R.SizeRep = R.Size(1:end-1);
            R.NTot = prod(R.Size);
            R.NRep = prod(R.SizeRep);
            
            R.DeoxySignal = cell(R.Size);
            R.DeoxySignalIntra = cell(R.Size);
            R.DeoxySignalExtra = cell(R.Size);
            R.DeoxySignalVRS = cell(R.Size);
            R.OxySignal = cell(R.Size);
            R.OxySignalIntra = cell(R.Size); 
            R.OxySignalExtra = cell(R.Size);
            R.OxySignalVRS = cell(R.Size);
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PUBLIC METHODS:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods ( Access = public, Static = false )
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % OVERLOADED METHODS:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [varargout] = size(R, varargin)
            
            if nargout <= 1
                switch nargin
                    case 1
                        varargout{1} = R.Size;
                    case 2
                        m = varargin{1};
                        if ~iswholenumber(m); error('m must be a positive integer.'); end
                        if m > R.NArgs
                            varargout{1} = 1;
                        else
                            varargout{1} = R.Size(m);
                        end
                    otherwise
                        error('Too many input arguments');
                end
            else
                if nargin > 1
                    error('Too many output arguments.');
                end
                varargout = cell(1,nargout);
                for ii = 1:min(nargout,R.NArgs)
                    varargout{ii} = R.Size(ii);
                end
                for ii = R.NArgs+1:nargout
                    varargout{ii} = 1;
                end
            end
            
        end
        
        function [inds] = find(R, varargin)
            switch numel(varargin)
                case 1
                    args = getargs(varargin{1});
                case R.NArgs
                    args = varargin;
                otherwise
                    error('Incorrect number of arguments');
            end
            
            Args  = getargs(R);
            idx   = cell(1,R.NArgs);
            for ii = 1:R.NArgs
                %[~,idx{ii},~] = intersect(Args{ii}(:),args{ii}(:));
                %[~,idx{ii},~] = intersectabstol(Args{ii}(:),args{ii}(:),1e-12);
                [~,idx{ii},~] = intersectreltol(Args{ii}(:),args{ii}(:),1e-12);
            end
            
            SizeSubs = cellfun(@numel,idx);
            NSubs = prod(SizeSubs);
            inds = cell(NSubs,R.NArgs);
            
            ix = cell(1,R.NArgs);
            for ii = 1:NSubs
                [ix{:}] = ind2sub(SizeSubs,ii);
                for jj = 1:R.NArgs
                    inds{ii,jj} = idx{1,jj}(ix{jj});
                end
            end
            
        end
        
        function [S] = mean(R)
            
            args = getargs(R);
            S = BOLDResults(args{1:end-1},1);
            
            if numel(R.Reps) == 1; return; end
             
            NumReps = numel(R.Reps);
            for ii = 1:NumReps
                for jj = 1:R.NRep
                    if ii == 1
                        S.OxySignal{jj} = [R.OxySignal{jj}(:,1), R.OxySignal{jj}(:,2)/NumReps];
                        S.OxySignalIntra{jj} = [R.OxySignalIntra{jj}(:,1), R.OxySignalIntra{jj}(:,2)/NumReps];
                        S.OxySignalExtra{jj} = [R.OxySignalExtra{jj}(:,1), R.OxySignalExtra{jj}(:,2)/NumReps];
                        S.OxySignalVRS{jj} = [R.OxySignalVRS{jj}(:,1), R.OxySignalVRS{jj}(:,2)/NumReps];
                        S.DeoxySignal{jj} = [R.DeoxySignal{jj}(:,1), R.DeoxySignal{jj}(:,2)/NumReps];
                        S.DeoxySignalIntra{jj} = [R.DeoxySignalIntra{jj}(:,1), R.DeoxySignalIntra{jj}(:,2)/NumReps];
                        S.DeoxySignalExtra{jj} = [R.DeoxySignalExtra{jj}(:,1), R.DeoxySignalExtra{jj}(:,2)/NumReps];
                        S.DeoxySignalVRS{jj} = [R.DeoxySignalVRS{jj}(:,1), R.DeoxySignalVRS{jj}(:,2)/NumReps];
                    else
                        S.OxySignal{jj}(:,2) = S.OxySignal{jj}(:,2) + R.OxySignal{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.OxySignalIntra{jj}(:,2) = S.OxySignalIntra{jj}(:,2) + R.OxySignalIntra{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.OxySignalExtra{jj}(:,2) = S.OxySignalExtra{jj}(:,2) + R.OxySignalExtra{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.OxySignalVRS{jj}(:,2) = S.OxySignalVRS{jj}(:,2) + R.OxySignalVRS{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.DeoxySignal{jj}(:,2) = S.DeoxySignal{jj}(:,2) + R.DeoxySignal{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.DeoxySignalIntra{jj}(:,2) = S.DeoxySignalIntra{jj}(:,2) + R.DeoxySignalIntra{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.DeoxySignalExtra{jj}(:,2) = S.DeoxySignalExtra{jj}(:,2) + R.DeoxySignalExtra{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                        S.DeoxySignalVRS{jj}(:,2) = S.DeoxySignalVRS{jj}(:,2) + R.DeoxySignalVRS{jj+(ii-1)*R.NRep}(:,2)/NumReps;
                    end
                end
            end
            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CUSTOM METHODS:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [echotimes, alphas, ...
                  boldsignals, boldsignalsIntra, boldsignalsExtra, boldsignalsVRS, ...
                  deoxysignals, deoxysignalsIntra, deoxysignalsExtra, deoxysignalsVRS, ...
                  oxysignals, oxysignalsIntra, oxysignalsExtra, oxysignalsVRS] = getBOLDSignals(R,varargin)
            
            numargs = getnumargs(R);
            if nargin > 1+numargs
                error('Too many input arguments');
            end
            args = getargs(R);
            
            echotimes = args{1}(:);
            alphas = args{2}(:);
            
            if nargin >= 2 && ~isempty(varargin{1}) %default is plot all if empty
                echotimes = varargin{1}(:);
            end
            
            if nargin >= 3 && ~isempty(varargin{2}) %default is plot all if empty
                alphas = varargin{2}(:);
            end
            
            args{1} = echotimes(:).';
            args{2} = alphas(:).';
            
            if nargin >= 4
                args(3:nargin-1) = varargin;
                for ii = 3:numargs
                    if ~isrow(args{ii})
                        args{ii} = reshape(args{ii},1,[]);
                    end
                end
            end
                        
            numOtherArgs = prod(cellfun(@numel,args(3:end)));
            boldsignals = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            boldsignalsIntra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            boldsignalsExtra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            boldsignalsVRS = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            deoxysignals = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            deoxysignalsIntra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            deoxysignalsExtra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            deoxysignalsVRS = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            oxysignals = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            oxysignalsIntra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            oxysignalsExtra = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            oxysignalsVRS = repmat({NaN(numel(echotimes),numel(alphas))},numOtherArgs,1);
            
            count = 0;
            sub = find(R,args{:});
            for idx = 1:numOtherArgs
                for jj = 1:numel(alphas)
                    for ii = 1:numel(echotimes)
                        count = count + 1;
                        s0 = R.DeoxySignal{sub{count,:}};
                        s0Intra = R.DeoxySignalIntra{sub{count,:}};
                        s0Extra = R.DeoxySignalExtra{sub{count,:}};
                        s0VRS = R.DeoxySignalVRS{sub{count,:}};
                        s = R.OxySignal{sub{count,:}};
                        sIntra = R.OxySignalIntra{sub{count,:}};
                        sExtra = R.OxySignalExtra{sub{count,:}};
                        sVRS = R.OxySignalVRS{sub{count,:}};
                        if ~isempty(s0) && ~isempty(s)
                            boldsignals{idx}(ii,jj) = abs(s(end,2)) - abs(s0(end,2));
                            boldsignalsIntra{idx}(ii,jj) = abs(sIntra(end,2)) - abs(s0Intra(end,2));
                            boldsignalsExtra{idx}(ii,jj) = abs(sExtra(end,2)) - abs(s0Extra(end,2));
                            boldsignalsVRS{idx}(ii,jj) = abs(sVRS(end,2)) - abs(s0VRS(end,2));
                            deoxysignals{idx}(ii,jj) = abs(s0(end,2));
                            deoxysignalsIntra{idx}(ii,jj) = abs(s0Intra(end,2));
                            deoxysignalsExtra{idx}(ii,jj) = abs(s0Extra(end,2));
                            deoxysignalsVRS{idx}(ii,jj) = abs(s0VRS(end,2));
                            oxysignals{idx}(ii,jj) = abs(s(end,2));
                            oxysignalsIntra{idx}(ii,jj) = abs(sIntra(end,2));
                            oxysignalsExtra{idx}(ii,jj) = abs(sExtra(end,2));
                            oxysignalsVRS{idx}(ii,jj) = abs(sVRS(end,2));
                        end
                    end
                end
            end
            
        end
        
        function R = fillwithmockdata(R)
            disp("fillwithmockdata");
            for ii = 1:R.NTot
                N = 2 * (4 + randi(8));
                e = 2e-2;
                
                dt = 2.5e-3;
                Time = (0:N).'*dt;
                
                R2 = 50.0 * (1 + e * randn(N,1));
                Phase = 200 * (1 + e * randn(N/2,1)) .* Time(1:N/2);
                Phase = [Phase; -fliplr(Phase) + 100 * (e * randn(N/2,1)) .* dt];
                Signal = [1; exp(-(R2+1i*Phase).*Time(2:end))];
                
                R2 = 31.1 * (1 + e * randn(N,1));
                Phase = 100 * (1 + e * randn(N/2,1)) .* Time(1:N/2);
                Phase = [Phase; -fliplr(Phase) + 100 * (e * randn(N/2,1)) .* dt];
                Signal0 = [1; exp(-(R2+1i*Phase).*Time(2:end))];
                
                R.DeoxySignal{ii} = [Time, Signal0];
                R.OxySignal{ii} = [Time, Signal];
            end
            
        end
        
        function args = getargs(R)
            args = { R.EchoTimes, R.Alphas, R.DeoxyBloodLevels, R.OxyBloodLevels, R.Hcts, R.Reps };
        end
        
        function num = getnumargs(R)
            num = R.NArgs;
        end
                
    end
    
end

