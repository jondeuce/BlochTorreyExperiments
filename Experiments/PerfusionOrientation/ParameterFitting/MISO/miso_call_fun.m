function y = miso_call_fun(fun, x)
%MISO_CALL_FUN y = miso_call_fun(fun, x)
% Wrap a call to `fun(x)` with MISO interface, updating global settings
% `sampledata`, as well as timing the function call.

% Global variable collecting sample points, function values, and evaluation times
global sampledata

x = x(:)'; % make sure vector is row vector
fevalt = tic; %start timer for function evaluation
y = fun(x); % call function
t = toc(fevalt); %stop timer for function evaluation

% Update sample data (point x, value y, time t)
sampledata = [sampledata; x, y, t];

end