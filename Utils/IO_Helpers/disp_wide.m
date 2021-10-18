function [varargout] = disp_wide(A, repzeroes, varargin)
%DISP_WIDE Displays the matrix A without wrapping. `repzeroes` replaces '0'
%with ' ' everywhere. `varargin` is forwarded to `num2str`.

if nargin < 2; repzeroes = false; end

S = num2str(A, varargin{:});
if repzeroes; S = reshape(strrep(S(:).', '0', ' ').', size(S)); end

fprintf('\n');
for ii = 1:size(S,1)
    fprintf('%s\n', S(ii,:));
end
fprintf('\n');

if nargout > 0
    varargout{1} = S;
end

end

