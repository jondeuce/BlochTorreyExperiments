function all_tests_passed = test(Gsize)

rng('default')
b = true;

if nargin < 1; Gsize = [7,6,5]; end
% if nargin < 1; Gsize = [5,3,3]; end
% if nargin < 1; Gsize = [4,4,4]; end
Vsize = (1+rand())*(Gsize./max(Gsize));
h = mean(Vsize./Gsize);
x0 = randnc(Gsize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Empty mask (for scalar D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mask = logical([]);

% Gamma randnc, Dcoeff scalar positive
Gamma = randnc(Gsize)/10;
Dcoeff = rand()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff scalar positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff scalar negative
Gamma = randnc(Gsize)/10;
Dcoeff = -rand()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff scalar negative', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff randnc
Gamma = randnc(Gsize)/10;
Dcoeff = randnc()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff == 0
Gamma = randnc(Gsize)/10;
Dcoeff = 0;
b = run_suite_combinations('Gamma randnc, Dcoeff == 0', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma scalar randnc, Dcoeff randnc
Gamma = randnc()/10;
Dcoeff = randnc()*h^2;
b = run_suite_combinations('Gamma scalar randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma scalar randnc, Dcoeff positive
Gamma = randnc()/10;
Dcoeff = rand()*h^2;
b = run_suite_combinations('Gamma scalar randnc, Dcoeff positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Identity test: Gamma == -1, Dcoeff == 0
Gamma = -1;
Dcoeff = 0;
b = run_suite_combinations('Gamma == -1, Dcoeff == 0', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variable D, empty mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mask = logical([]);

% Gamma zeros, Dcoeff h^2 * ones array
Gamma = zeros(Gsize);
Dcoeff = h^2 * ones(Gsize);
b = run_suite_combinations('Gamma zeros, Dcoeff h^2 * ones array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma zeros, Dcoeff const array
Gamma = zeros(Gsize);
Dcoeff = rand()*ones(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff const array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff zeros array
Gamma = randnc(Gsize)/10;
Dcoeff = zeros(Gsize);
b = run_suite_combinations('Gamma randnc, Dcoeff zeros array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma zeros, Dcoeff randn array
Gamma = zeros(Gsize);
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff randn array
Gamma = randnc(Gsize)/10;
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff randnc array
Gamma = randnc(Gsize)/10;
Dcoeff = randnc(Gsize)*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randnc array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variable D, non-empty mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mask = rand(Gsize) > 0.5; % Random mask
% mask = double(mask);
mask = randi([0,3], Gsize);
% [X,Y,Z] = meshgrid(...
%     linspace(-Vsize(2)/2, Vsize(2)/2, Gsize(2)), ...
%     linspace(-Vsize(1)/2, Vsize(1)/2, Gsize(1)), ...
%     linspace(-Vsize(3)/2, Vsize(3)/2, Gsize(3)));
% R = min(Vsize(:))/4;
% mask = X.^2 + Y.^2 + Z.^2 <= R^2; % Spherical mask

% Gamma zeros, Dcoeff h^2 * ones array
Gamma = zeros(Gsize);
Dcoeff = h^2 * ones(Gsize);
b = run_suite_combinations('Gamma zeros, Dcoeff h^2 * ones array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma zeros, Dcoeff const array
Gamma = zeros(Gsize);
Dcoeff = rand()*ones(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff const array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff zeros array
Gamma = randnc(Gsize)/10;
Dcoeff = zeros(Gsize);
b = run_suite_combinations('Gamma randnc, Dcoeff zeros array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma zeros, Dcoeff randn array
Gamma = zeros(Gsize);
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff randn array
Gamma = randnc(Gsize)/10;
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Gamma randnc, Dcoeff randnc array
Gamma = randnc(Gsize)/10;
Dcoeff = randnc(Gsize)*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randnc array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize, mask) && b;

% Finish up
if b; fprintf('\nAll tests passed\n\n');
else; warning('Some tests failed');
end

if nargout > 0; all_tests_passed = b; end

end

function b = run_suite_combinations(name, x0, Gamma, Dcoeff, Gsize, Vsize, mask)

% run all (necessary) combinations of real/complex inputs
[xreal, greal, dreal] = deal(isreal(x0), isreal(Gamma), isreal(Dcoeff));
[c1,c2,c3,c4,c5,c6,c7,c8] = deal(true, ...
    ~xreal, ~greal, ~dreal, ...
    ~xreal && ~greal, ~xreal && ~dreal, ~greal && ~dreal, ...
    ~xreal && ~greal && ~dreal);

b = true;
if c1; b = run_suite(name, x0,       Gamma,       Dcoeff,       Gsize, Vsize, mask) && b; end
if c2; b = run_suite(name, real(x0), Gamma,       Dcoeff,       Gsize, Vsize, mask) && b; end
if c3; b = run_suite(name, x0,       real(Gamma), Dcoeff,       Gsize, Vsize, mask) && b; end
if c4; b = run_suite(name, x0,       Gamma,       real(Dcoeff), Gsize, Vsize, mask) && b; end
if c5; b = run_suite(name, real(x0), real(Gamma), Dcoeff,       Gsize, Vsize, mask) && b; end
if c6; b = run_suite(name, real(x0), Gamma,       real(Dcoeff), Gsize, Vsize, mask) && b; end
if c7; b = run_suite(name, x0,       real(Gamma), real(Dcoeff), Gsize, Vsize, mask) && b; end
if c8; b = run_suite(name, real(x0), real(Gamma), real(Dcoeff), Gsize, Vsize, mask) && b; end

end

function b = run_suite(name, x0, Gamma, Dcoeff, Gsize, Vsize, mask)

boolchoose = @(b,x,y) b.*x + (1-b).*y; % returns `x` if `b` is true, `y` otherwise
% randstate = @() boolchoose(randi(0:1), BlochTorreyOp.DiagState, BlochTorreyOp.GammaState);
% randstate = @() BlochTorreyOp.GammaState;
% randstate = @() BlochTorreyOp.DiagState;

Ns = 45; % string message pad length
b = true;
isdiag = false;
A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize, isdiag, mask);

As = sparse(A);
Af = full(A);
Ab = full_Brute(Gamma, Dcoeff, Gsize, Vsize, mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test state switching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = setbuffer(A, BlochTorreyOp.GammaState);
b = test_approx_eq(full(A), full(setbuffer(A, BlochTorreyOp.DiagState)), ...
    name, strpad('GammaState to DiagState switching equal', Ns)) && b;
A = setbuffer(A, BlochTorreyOp.DiagState);
b = test_approx_eq(full(A), full(setbuffer(A, BlochTorreyOp.GammaState)), ...
    name, strpad('DiagState to GammaState switching equal', Ns)) && b;

% for state = [BlochTorreyOp.GammaState]
% for state = [BlochTorreyOp.DiagState]
for state = [BlochTorreyOp.GammaState, BlochTorreyOp.DiagState]
    
    A  = setbuffer(A, state);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Matrix multiplication testing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Matrix multiplication
    yb = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize, mask);
    y  = A*x0;
    ys = reshape(As*vec(x0), size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop mat*vec', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse mat*vec', Ns)) && b;

    % Matrix-transpose multiplication
    yb = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize, mask); % symmetric
    y  = A.'*x0;
    ys = reshape(As.'*vec(x0), size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop trans-mat*vec', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse trans-mat*vec', Ns)) && b;

    % Matrix-conjugate-transpose multiplication
    yb = BlochTorreyBrute(x0, conj(Gamma), conj(Dcoeff), Gsize, Vsize, mask);
    y  = A'*x0;
    ys = reshape(As'*vec(x0), size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop conj-trans-mat*vec', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse conj-trans-mat*vec', Ns)) && b;

    % Vector*Matrix multiplication (3D grid)
    yb = BlochTorreyBrute(conj(x0), Gamma, Dcoeff, Gsize, Vsize, mask);
    y  = conj(x0)*A;
    ys = reshape(vec(x0)'*As, size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop vec-ctrans*mat (3D)', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse vec-ctrans*mat (3D)', Ns)) && b;

    % Vector*Matrix multiplication (1D vector)
    yb = BlochTorreyBrute(conj(x0), Gamma, Dcoeff, Gsize, Vsize, mask);
    y  = vec(x0)'*A;
    ys = vec(x0)'*As;

    b = test_approx_eq(yb, y, name, strpad('BTop vec-ctrans*mat (1D)', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse vec-ctrans*mat (1D)', Ns)) && b;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Matrix properties testing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    b = test_approx_eq(Ab, Af, name, strpad('BTop matrix equal', Ns)) && b;
    b = test_approx_eq(Ab, full(As), name, strpad('BTsparse matrix equal', Ns)) && b;
    % b = test_approx_eq(abs(Ab), full(abs(A)), name, strpad('BTop abs equal', Ns)) && b;
    b = test_approx_eq(real(Ab), full(real(A)), name, strpad('BTop real equal', Ns)) && b;
    b = test_approx_eq(imag(Ab), full(imag(A)), name, strpad('BTop imag equal', Ns)) && b;
    b = test_approx_eq(diag(Ab), diag(A), name, strpad('BTop diag equal', Ns)) && b;
    b = test_approx_eq(norm(Ab,1), norm(A,1), name, strpad('BTop 1-norm equal', Ns)) && b;
    b = test_approx_eq(norm(Ab,inf), norm(A,inf), name, strpad('BTop inf-norm equal', Ns)) && b;
    b = test_approx_eq(norm(Ab,'fro'), norm(A,'fro'), name, strpad('BTop frob-norm equal', Ns), 100) && b; % more roundoff errors from squaring

    o = offdiagonals(A); if size(o,1) == 1; o = repmat(o, length(A), 1); end
    ob = get_offdiagonals(Ab, Gsize);
    b = test_approx_eq(ob, o, name, strpad('BTop offdiagonals equal', Ns), 5*sqrt(length(A))) && b;
        
    % trace is sensitive to floating point arithmetic; error should be O(sqrt(N))*eps
    b = test_approx_eq(trace(Ab), trace(A), name, strpad('BTop trace equal', Ns), 5*sqrt(length(A))) && b;

    D = diag(randnc(length(A),1));
    b = test_approx_eq(Ab+D, full(A+D), name, strpad('BTop + diag mat equal', Ns)) && b;
    b = test_approx_eq(Ab+D, full(D+A), name, strpad('diag mat + BTop equal', Ns)) && b;
    b = test_approx_eq(2*Ab, full(A+A), name, strpad('BTop + BTop mat equal', Ns)) && b;
    b = test_approx_eq(zeros(size(A)), full(A-A), name, strpad('BTop - BTop mat equal zeros', Ns)) && b;
    
    c1 = exp(2i*pi*rand()); c2 = exp(2i*pi*rand());
    b = test_approx_eq((c1+c2)*Ab, full(c1*A+c2*A), name, strpad('BTop linear combination c1*A+c2*A', Ns), 10) && b; % more roundoff errors from complex cancellation?
    b = test_approx_eq((c1-c2)*Ab, full(c1*A-c2*A), name, strpad('BTop linear combination c1*A-c2*A', Ns), 10) && b; % more roundoff errors from complex cancellation?
    
    mu = trace(Ab)/length(Ab);
    b  = test_approx_eq(Ab-mu*eye(size(Ab),'like',Ab), full(A-mu*eye(size(A),'like',A)), name, strpad('BTop - mu*I mat equal', Ns), 10) && b; % more roundoff errors from complex cancellation?
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Scalar multiplication testing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Real-scalar multiplication
    a  = randn();
    yb = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize, mask);
    y  = (a*A)*x0;
    ys = reshape((a*As)*vec(x0), size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop RHS-real-scalar*mat*vec', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse RHS-real-scalar*mat*vec', Ns)) && b;

    y = (A*a)*x0;
    b = test_approx_eq(yb, y, name, strpad('BTop LHS-real-scalar*mat*vec', Ns)) && b;
    b = test_approx_eq(yb, ys, name, strpad('BTsparse LHS-real-scalar*mat*vec', Ns)) && b;

    % Complex-scalar multiplication
    a  = randnc();
    yb = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize, mask);
    y  = (a*A)*x0;
    ys = reshape((a*As)*vec(x0), size(x0));

    b = test_approx_eq(yb, y, name, strpad('BTop RHS-cplx-scalar*mat*vec', Ns), 10) && b; % more roundoff errors from complex cancellation?
    b = test_approx_eq(yb, ys, name, strpad('BTsparse RHS-cplx-scalar*mat*vec', Ns), 10) && b; % more roundoff errors from complex cancellation?

    y = (A*a)*x0;
    b = test_approx_eq(yb, y, name, strpad('BTop LHS-cplx-scalar*mat*vec', Ns), 10) && b; % more roundoff errors from complex cancellation?
    b = test_approx_eq(yb, ys, name, strpad('BTsparse LHS-cplx-scalar*mat*vec', Ns), 10) && b; % more roundoff errors from complex cancellation?

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Exponential matrix-vector product testing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    t  = 0.1 * rand();
    yb = expm(t*Ab) * vec(x0);
    V  = ExpmvStepper(t, A, [], [], 'prnt', false);
    V  = precompute(V, x0);
    y  = step(V, x0);
    % y  = bt_expmv(t, A, x0, 'prnt', false);
    ys = expmv(t, As, vec(x0), [], 'double', true, false, false, false);
    b  = test_approx_eq(yb, y, name, strpad('BTop expmv equal (GRE)', Ns), 100) && b;
    b  = test_approx_eq(ys, y, name, strpad('BTsparse expmv equal (GRE)', Ns), 100) && b;

    t  = 0.1 * rand();
    Ef = expm(t/2*Ab);
    yb = Ef * conj( Ef * vec(x0) );
    % V  = ExpmvStepper(t/2, A, [], [], 'prnt', false, 'type', 'GRE');
    % V  = precompute(V, x0);
    % [y,~,~,V]  = step(V, x0);
    % y = step(V, conj(y));
    V  = ExpmvStepper(t, A, [], [], 'prnt', false, 'type', 'SE');
    V  = precompute(V, x0);
    y  = step(V, x0);
    ys = expmv(t/2, As, vec(x0), [], 'double', true, false, false, false);
    ys = expmv(t/2, As, conj(ys), [], 'double', true, false, false, false);
    b  = test_approx_eq(yb, y, name, strpad('BTop expmv equal (SE)', Ns), 100) && b;
    b  = test_approx_eq(ys, y, name, strpad('BTsparse expmv equal (SE)', Ns), 100) && b;
    
end % state loop

end

function y = BlochTorreyBrute(x0, Gamma, D, Gsize, Vsize, m)
h = mean(Vsize./Gsize);
y = BlochTorreyAction_brute(x0, h, D, Gamma, 1, false, m);
end

function A = full_Brute(Gamma, D, Gsize, Vsize, m)
N = prod(Gsize);
A = zeros(N,N);
ei = zeros(Gsize);

ei(1) = 1;
A(:,1) = vec(BlochTorreyBrute(ei, Gamma, D, Gsize, Vsize, m));
for ii = 2:size(A,2)
    ei(ii-1) = 0;
    ei(ii) = 1;
    A(:,ii) = vec(BlochTorreyBrute(ei, Gamma, D, Gsize, Vsize, m));
end

end

function A = full_BTAction(Gamma, D, Gsize, Vsize, mask)
N = prod(Gsize);
h = mean(Vsize./Gsize);
Gamma = complex(Gamma);

A = zeros(N,N);
ei = complex(zeros(Gsize));

ei(1) = 1;
A(:,1) = vec(BlochTorreyAction(ei, h, D, Gamma, Gsize, 1, false, false));
for ii = 2:size(A,2)
    ei(ii-1) = 0;
    ei(ii) = 1;
    A(:,ii) = vec(BlochTorreyAction(ei, h, D, Gamma, Gsize, 1, false, false));
end

end

function p = strpad(s,N)
p = repmat(' ',1,N);
p(1:length(s)) = s;
end

function str = errmsg(name, msg)
str = sprintf('Test failed: %s (test suite: %s)', msg, name);
end

function str = passedmsg(name, msg)
str = sprintf('Test passed: %s (test suite: %s)', msg, name);
end

function b = test_approx_eq(x,y,name,msg,tolfact)
if nargin < 5; tolfact = 5; end
if nargin < 4; msg = 'test failed'; end
if nargin < 3; name = 'N/A'; end

maxval = max(infnorm(x), infnorm(y));
tol    = max(tolfact * eps(maxval), tolfact * eps);
% tol  = sqrt(tol); % For testing if failures are due to tolerance

maxdiff = max(abs(vec(x)-vec(y)));
b = (maxdiff <= tol);

if b
    fprintf('%s\n',passedmsg(name,msg));
else
    warning(errmsg(name,msg));
    if isscalar(x) && isscalar(y)
        fprintf('val1: '); disp(x);
        fprintf('val2: '); disp(y);
    else
        fprintf('max error: '); disp(maxdiff);
        fprintf('tolerance: '); disp(tol);
        fprintf('max value: '); disp(maxval);
    end
end

end

function o = get_offdiagonals(A, Gsize)

mod1X = @(x) 1 + mod(x-1, Gsize(1)); % circular indexing in x/y/z-direction
mod1Y = @(x) 1 + mod(x-1, Gsize(2));
mod1Z = @(x) 1 + mod(x-1, Gsize(3));

iC = vec(1:prod(Gsize)); % row indices of center cell

[I,J,K] = ind2sub(Gsize, iC);
iD = sub2ind(Gsize, mod1X(I-1), J, K); % column indices of 6 neighbouring cells
iU = sub2ind(Gsize, mod1X(I+1), J, K);
iL = sub2ind(Gsize, I, mod1Y(J-1), K);
iR = sub2ind(Gsize, I, mod1Y(J+1), K);
iB = sub2ind(Gsize, I, J, mod1Z(K-1));
iF = sub2ind(Gsize, I, J, mod1Z(K+1));

iD = sub2ind(size(A), iC, iD); % linear indices of 6 neighbouring cells
iU = sub2ind(size(A), iC, iU);
iL = sub2ind(size(A), iC, iL);
iR = sub2ind(size(A), iC, iR);
iB = sub2ind(size(A), iC, iB);
iF = sub2ind(size(A), iC, iF);

o = [A(iD), A(iU), A(iL), A(iR), A(iB), A(iF)];

end
