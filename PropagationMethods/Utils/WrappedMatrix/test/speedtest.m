function [ passed ] = speedtest( func, nreps, A, B )
%SPEEDTEST Summary of this function goes here
%   Detailed explanation goes here

if nargin == 3
    test_passed = unary_speedtest( func, nreps, A );
elseif nargin == 4
    test_passed = binary_speedtest( func, nreps, A, B );
else
    error('Only unary and binary tests');
end

if nargout > 0
    passed = test_passed;
end

end

function passed = unary_speedtest( func, nreps, A )

sfunc = func2str(func);
f = WrappedMatrix(A);

B = func(A);
g = func(f);

if maxabs(B-g) > 10 * eps(class(B))
    error('Unary Op: %s(A) ~= %s(f)', sfunc);
else
    passed = true;
end

tic;
for ii = 1:nreps
    B = func(A);
end
StandardTime = toc/nreps;

tic;
for ii = 1:nreps
    g = func(f);
end
WrappedTime = toc/nreps;

display_toc_time(StandardTime, [sfunc, ': Standard']);
display_toc_time(WrappedTime,  [sfunc, ': Wrapped ']);
fprintf('Wrapped / Standard: %s\n', num2str(WrappedTime/StandardTime,3));
fprintf('Wrapped - Standard: %s ms\n', num2str(1000 * (WrappedTime-StandardTime),3));

end

function passed = binary_speedtest( func, nreps, A, B )

sfunc = func2str(func);
f = WrappedMatrix(A);
g = WrappedMatrix(B);

C = func(A,B);
h = func(f,g);

if maxabs(C-h) > 10 * eps(class(C))
    error('Binary Op: %s(A) ~= %s(f)', sfunc);
else
    passed = true;
end

tic;
for ii = 1:nreps
    C = func(A,B);
end
StandardTime = toc/nreps;

tic;
for ii = 1:nreps
    h = func(f,g);
end
WrappedTime = toc/nreps;

display_toc_time(StandardTime, [sfunc, ': Standard']);
display_toc_time(WrappedTime,  [sfunc, ': Wrapped ']);
fprintf('Wrapped / Standard: %s\n', num2str(WrappedTime/StandardTime,3));
fprintf('Wrapped - Standard: %s ms\n', num2str(1000 * (WrappedTime-StandardTime),3));

end