function R = push(R, varargin)
%PUSH Push signal into R

switch numel(varargin)
    case 1
        S = varargin{1};
        args = getargs(S);
        Signal0 = S.DeoxySignal;
        Signal0_Intra = S.DeoxySignalIntra;
        Signal0_Extra = S.DeoxySignalExtra; 
        Signal0_VRS = S.DeoxySignalVRS; 
        Signal = S.OxySignal;
        Signal_Intra = S.OxySignalIntra;
        Signal_Extra = S.OxySignalExtra;
        Signal_VRS = S.OxySignalVRS;
        
    %case R.NArgs+2    
    case R.NArgs+8  % +2 for Intravascular +2 Extravascullar +2 VRS
        
        %Signal0 and Signal should be Nx2 arrays with the first
        %column time and the second column the complex signal,
        %or cell arrays of such Nx2 arrays
        
        if isempty(varargin{1})
            Signal0 = [];
        else
            Signal0 = varargin(1);
            if iscell(Signal0{1}); Signal0 = Signal0{1}; end
        end
        
        if isempty(varargin{2})
            Signal0_Intra = [];
        else
            Signal0_Intra = varargin(2);
            if iscell(Signal0_Intra{1}); Signal0_Intra = Signal0_Intra{1}; end
        end
        
        if isempty(varargin{3})
            Signal0_Extra = [];
        else
            Signal0_Extra = varargin(3);
            if iscell(Signal0_Extra{1}); Signal0_Extra = Signal0_Extra{1}; end
        end
        
        if isempty(varargin{4})
            Signal0_VRS = [];
        else
            Signal0_VRS = varargin(4);
            if iscell(Signal0_VRS{1}); Signal0_VRS = Signal0_VRS{1}; end
        end
        
        if isempty(varargin{5})
            Signal = [];
        else
            Signal = varargin(5);
            if iscell(Signal{1});  Signal  = Signal{1};  end
        end
        
        if isempty(varargin{6})
            Signal_Intra = [];
        else
            Signal_Intra = varargin(6);
            if iscell(Signal_Intra{1});  Signal_Intra  = Signal_Intra{1};  end
        end
        
        if isempty(varargin{7})
            Signal_Extra = [];
        else
            Signal_Extra = varargin(7);
            if iscell(Signal_Extra{1});  Signal_Extra  = Signal_Extra{1};  end
        end
        
        if isempty(varargin{8})
            Signal_VRS = [];
        else
            Signal_VRS = varargin(8);
            if iscell(Signal_VRS{1});  Signal_VRS  = Signal_VRS{1};  end
        end
        
        args = varargin(9:end);
        
    otherwise
        error('Incorrect number of arguments');
end

inds = find(R, args{:}); %echotimes

for ii = 1:size(inds,1)
    if ~isempty(Signal0)
        R.DeoxySignal{inds{ii,:}} = Signal0{ii};
    end
    if ~isempty(Signal0_Intra)
        R.DeoxySignalIntra{inds{ii,:}} = Signal0_Intra{ii};
    end
    if ~isempty(Signal0_Extra)
        R.DeoxySignalExtra{inds{ii,:}} = Signal0_Extra{ii};
    end
    if ~isempty(Signal0_VRS)
        R.DeoxySignalVRS{inds{ii,:}} = Signal0_VRS{ii};
    end
    if ~isempty(Signal)
        R.OxySignal{inds{ii,:}} = Signal{ii};
    end
    if ~isempty(Signal_Intra)
        R.OxySignalIntra{inds{ii,:}} = Signal_Intra{ii};
    end
    if ~isempty(Signal_Extra)
        R.OxySignalExtra{inds{ii,:}} = Signal_Extra{ii};
    end
    if ~isempty(Signal_VRS)
        R.OxySignalVRS{inds{ii,:}} = Signal_VRS{ii};
    end
end

end