function [y,t] = BTSplitStep(V, x, t0, D, Gamma, verb)
%BTSplitStep [y,t] = BTSplitStep(V, x, t0, D, Gamma, verb)

if nargin < 6; verb = false; end
if nargin < 5; Gamma = []; end
if nargin < 4; D = []; end
if nargin < 3; t0 = 0; end

if ~isempty(D); V = precomputeConvKernels(V,D); end
if ~isempty(Gamma); V = precomputeExpDecays(V,Gamma); end

switch V.Order
    case 2
        %TODO: [y,t] = BTSplitStepOrder2(V, x0, t0, verb);
        [y,t] = BTSplitStepOrderN(V, x, t0, verb);
    case 4
        %TODO: [y,t] = BTSplitStepOrder4(V, x0, t0, verb);
        [y,t] = BTSplitStepOrderN(V, x, t0, verb);
    otherwise
        [y,t] = BTSplitStepOrderN(V, x, t0, verb);
end

end

function [y,t] = BTSplitStepOrder2(V, x, t0, verb)
error('not impl.'); %TODO
end

function [y,t] = BTSplitStepOrder4(V, x, t0, verb)
error('not impl.'); %TODO
end

function [y,t] = BTSplitStepOrderN(V, x, t0, verb)

% 'Nreps' timesteps of length 'TimeStep' are taken from t0
t = t0 + V.NReps * V.TimeStep;

% Number of multiplication steps is 2N, convolution steps 2N-1, for a
% total of 4N-1 steps
N = length(V.b);

% Convenience functions
exporder = [1:N, N:-1:1];
expstep = @(step) exporder(ceil(step/2));
convorder = [1:N, N-1:-1:1];
convstep = @(step) convorder(round(step/2));

expprnt = @(step,n) verb && fprintf('step %d\t\texp  %d\n', step, n);
convprnt = @(step,m) verb && fprintf('step %d\t\tconv %d\n', step, m);
chainprnt = @(nsub) verb && fprintf('step %d & %d\tchain %d\n', 4*N-1, 1, nsub-1);

% Start (step 1)
y = V.ExpArrays{1} .* x;
expprnt(1,1);

% Inner loop (steps 2 to 4N-2)
y = conv(V.ConvKernels{1}, y);
convprnt(2,1);
for step = 3:2:4*N-2
    [n,m] = deal(expstep(step), convstep(step+1));
    y = V.ExpArrays{n} .* y;
    y = conv(V.ConvKernels{m}, y);
    expprnt(step,n);
    convprnt(step+1,m);
end

for nsub = 2:V.NReps
    % Chain step (previous step 4N-1 and the next step 1)
    y = V.ExpChain .* y;
    chainprnt(nsub);
    
    % Step 2
    y = conv(V.ConvKernels{1}, y);
    convprnt(2,1);
    
    % Inner loop (steps 3 to 4N-2)
    for step = 3:2:4*N-2
        [n,m] = deal(expstep(step), convstep(step+1));
        
        y = V.ExpArrays{n} .* y;
        y = conv(V.ConvKernels{m}, y);
        
        expprnt(step,n);
        convprnt(step+1,m);
    end
end

% End (step 4N-1)
y = V.ExpArrays{1} .* y;
expprnt(4*N-1,1);

end
