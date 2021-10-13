function varargout = trans( r0, varargin )
% translates m x n matrices by the amount specified in the n x 1
% vector 'r0'
% 
%   e.g.    [A,B,C]=trans( r0, A, B, C ) translates the m x n matrices
%           A, B, and C by the n x 1 vector r0

r0=r0(:);
N=length(r0);
M=length(varargin);

if norm(r0)<1e-6;
    for k=1:M; varargout{k}=varargin{k}; end;
    return;
end

for ind=1:M
    for n=1:N
        varargout{ind}(:,n)=varargin{ind}(:,n)-r0(n);
    end
end

end

