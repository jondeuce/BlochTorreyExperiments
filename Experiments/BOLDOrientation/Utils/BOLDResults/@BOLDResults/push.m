function R = push(R, varargin)
%PUSH Push signal into R

switch numel(varargin)
    case 1
        S = varargin{1};
        args = getargs(S);
        Signal0 = S.DeoxySignal;
        Signal = S.OxySignal;
        
    case R.NArgs+2
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
            Signal = [];
        else
            Signal = varargin(2);
            if iscell(Signal{1});  Signal  = Signal{1};  end
        end
        
        args = varargin(3:end);
        
    otherwise
        error('Incorrect number of arguments');
end

inds = find(R, args{:});
for ii = 1:size(inds,1)
    if ~isempty(Signal0)
        R.DeoxySignal{inds{ii,:}} = Signal0{ii};
    end
    if ~isempty(Signal)
        R.OxySignal{inds{ii,:}} = Signal{ii};
    end
end

end