function [ T ] = merge_nonempty( R, S )
%MERGE_NONEMPTY Merge R and S, pushing R into new object T and keeping its
% MetaData, and pushing non-empty signals from S into R afterwards.

% Find common arguments
Targs = getcommonargs(R,S);
T = BOLDResults(Targs{:});

% Push R results in
T.MetaData = R.MetaData;
T = push(T,R);

% Manually push S results, if they are non-empty
Sargs = getargs(S);
Signal0 = S.DeoxySignal;
Signal = S.OxySignal;

inds = find(T, Sargs{:});
for ii = 1:size(inds,1)
    if ~isempty(Signal0) && ~isempty(Signal0{ii})
        T.DeoxySignal{inds{ii,:}} = Signal0{ii};
    end
    if ~isempty(Signal) && ~isempty(Signal{ii})
        T.OxySignal{inds{ii,:}} = Signal{ii};
    end
end

end

