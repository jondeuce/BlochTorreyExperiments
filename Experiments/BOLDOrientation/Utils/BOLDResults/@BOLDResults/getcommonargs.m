function args = getcommonargs(R,S,reltol)
%GETCOMMONARGS Returns the common arguments between R and S, up to a
%relative tolerance RELTOL (default 1e-12).

if nargin < 3; reltol = 1e-12; end

Sargs = getargs(S);
Rargs = getargs(R);

args = cell(size(Sargs));
for ii = 1:numel(args)
    args{ii} = uniquereltol([Sargs{ii}(:); Rargs{ii}(:)], reltol).';
end

end