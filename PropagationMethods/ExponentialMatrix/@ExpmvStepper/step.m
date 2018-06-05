function [y,dy,t] = step(V, x, dx, t0, computederivs)
%STEP [y,dy,t] = step(V, x, dx, t0, varargin)

if nargin < 5; computederivs = false; end
if nargin < 4; t0 = 0; end
if nargin < 3; dx = {};
elseif isempty(dx); if ~iscell(dx); dx = {}; end % convert [] to {}
elseif ~iscell(dx); dx = {dx}; % wrap in cell
end

if computederivs
    error('Derivative computation is not implemented for expmv stepping');
end

% ---- Precompute, if necessary ---- %
if ~V.isprecomputed; V = precompute(V, x); end

% ---- Step solution ---- %
y = bt_expmv( V.TimeStep, V.A, x, V.opts, V.selectdegargs, V.expmvargs );
t = t0 + V.TimeStep;

% ---- Step derivative ---- %
dy = {}; %TODO

end
