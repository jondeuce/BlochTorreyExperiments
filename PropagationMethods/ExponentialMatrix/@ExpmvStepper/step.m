function [y,dy,t,V] = step(V, x, dx, t0, computederivs)
%STEP [y,dy,t,V] = step(V, x, dx, t0, computederivs)

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
if isdiag(V.A)
    E = exp(V.TimeStep * diag(V.A));
    y = reshape(x, [size(V.A,2), numel(x)/size(V.A,2)]);
    y = bsxfun(@times, E, y);
    y = reshape(y, size(x));
    m_min = [];
else
    [y, ~, ~, m_min] = bt_expmv(V.TimeStep, V.A, x, V.opts, V.selectdegargs, V.expmvargs);
end
t = t0 + V.TimeStep;

% ---- Step derivative ---- %
dy = {}; %TODO

% ---- Update stepper ---- %
if V.adapttaylor; V = update_m_min(V, m_min); end

end

function V = update_m_min(V, m_min)

m_min_curr = V.expmvargs{end};
if isempty(m_min_curr)
    m_min_new = m_min;
else
    if isequal(size(m_min_curr), size(m_min))
        if isequal(m_min_curr, m_min)
            m_min_new = max(m_min_curr-1, 1);
        else
            m_min_new = min(m_min_curr, m_min);
        end
    else
        m_min_new = m_min;
    end
end
V.expmvargs{end} = m_min_new;

end