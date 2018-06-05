function h = checkGridSpacing(h)

if isempty(h)
    h = 1;
elseif ~(isfloat(h) && all(h > 0))
    error('h must be a positive floating point scalar or 3-element array');
end
if ~BlochTorreyOp.is_isotropic(h)
    error('h must be isotropic');
end

h = h(1);
if ~isa(h,'double'); h = double(h); end

end