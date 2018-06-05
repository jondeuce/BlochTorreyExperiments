function test

rng('default')
b = true;

Gsize = [5,6,4];
Vsize = (1+rand())*(Gsize./max(Gsize));
h = mean(Vsize./Gsize);
x0 = randnc(Gsize);

% Gamma randnc, Dcoeff scalar positive
Gamma = randnc(Gsize)/10;
Dcoeff = rand()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff scalar positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma randnc, Dcoeff scalar negative
Gamma = randnc(Gsize)/10;
Dcoeff = -rand()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff scalar negative', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma randnc, Dcoeff randnc
Gamma = randnc(Gsize)/10;
Dcoeff = randnc()*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma scalar randnc, Dcoeff randnc
Gamma = randnc()/10;
Dcoeff = randnc()*h^2;
b = run_suite_combinations('Gamma scalar randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma scalar randnc, Dcoeff positive
Gamma = randnc()/10;
Dcoeff = rand()*h^2;
b = run_suite_combinations('Gamma scalar randnc, Dcoeff positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Identity test: Gamma == -1, Dcoeff == 0
Gamma = -1;
Dcoeff = 0;
b = run_suite_combinations('Gamma == -1, Dcoeff == 0', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma zeros, Dcoeff randn array
Gamma = zeros(Gsize);
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma randnc, Dcoeff randn array
Gamma = randnc(Gsize)/10;
Dcoeff = randn(Gsize)*h^2;
b = run_suite_combinations('Gamma randnc, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Gamma zeros, Dcoeff const array
Gamma = zeros(Gsize);
Dcoeff = rand()*ones(Gsize)*h^2;
b = run_suite_combinations('Gamma zeros, Dcoeff const array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize) && b;

% Finish up
if b; fprintf('\nAll tests passed\n\n');
else; warning('Some tests failed');
end

end

function b = run_suite_combinations(name, x0, Gamma, Dcoeff, Gsize, Vsize)

% run all (necessary) combinations of real/complex inputs
[xreal, greal, dreal] = deal(isreal(x0), isreal(Gamma), isreal(Dcoeff));
[c1,c2,c3,c4,c5,c6,c7,c8] = deal(true, ...
    ~xreal, ~greal, ~dreal, ...
    ~xreal && ~greal, ~xreal && ~dreal, ~greal && ~dreal, ...
    ~xreal && ~greal && ~dreal);

b = true;
if c1; b = run_suite(name, x0,       Gamma,       Dcoeff,       Gsize, Vsize) && b; end
if c2; b = run_suite(name, real(x0), Gamma,       Dcoeff,       Gsize, Vsize) && b; end
if c3; b = run_suite(name, x0,       real(Gamma), Dcoeff,       Gsize, Vsize) && b; end
if c4; b = run_suite(name, x0,       Gamma,       real(Dcoeff), Gsize, Vsize) && b; end
if c5; b = run_suite(name, real(x0), real(Gamma), Dcoeff,       Gsize, Vsize) && b; end
if c6; b = run_suite(name, real(x0), Gamma,       real(Dcoeff), Gsize, Vsize) && b; end
if c7; b = run_suite(name, x0,       real(Gamma), real(Dcoeff), Gsize, Vsize) && b; end
if c8; b = run_suite(name, real(x0), real(Gamma), real(Dcoeff), Gsize, Vsize) && b; end

end

function b = run_suite(name, x0, Gamma, Dcoeff, Gsize, Vsize)

Ns = 35; % string message pad length
b = true;

A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize);
As = sparse(A);
Af = full(A);
Ab = full_Brute(Gamma, Dcoeff, Gsize, Vsize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix properties testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = test_approx_eq(Ab, Af, name, strpad('BTop matrix equal', Ns)) && b;
b = test_approx_eq(Ab, full(As), name, strpad('BTsparse matrix equal', Ns)) && b;
b = test_approx_eq(abs(Ab), full(abs(A)), name, strpad('BTop abs equal', Ns)) && b;
b = test_approx_eq(real(Ab), full(real(A)), name, strpad('BTop real equal', Ns)) && b;
b = test_approx_eq(imag(Ab), full(imag(A)), name, strpad('BTop imag equal', Ns)) && b;
b = test_approx_eq(diag(Ab), diag(A), name, strpad('BTop diag equal', Ns)) && b;
b = test_approx_eq(norm(Ab,1), norm(A,1), name, strpad('BTop 1-norm equal', Ns)) && b;
b = test_approx_eq(norm(Ab,inf), norm(A,inf), name, strpad('BTop inf-norm equal', Ns)) && b;
b = test_approx_eq(norm(Ab,'fro'), norm(A,'fro'), name, strpad('BTop frob-norm equal', Ns), 100) && b;

% trace is sensitive to floating point arithmetic; error should be O(sqrt(N))*eps
b = test_approx_eq(trace(Ab), trace(A), name, strpad('BTop trace equal', Ns), 5*sqrt(length(A))) && b;

% exponential matrix-vector product testing
t  = 0.1*rand();
yb = expm(t*Af)*x0(:);
V  = ExpmvStepper(t, A, [], [], 'prnt', false);
V  = precompute(V, x0);
y  = step(V, x0);
% y  = bt_expmv(t, A, x0, 'prnt', false);
ys = expmv(t, As, x0(:), [], 'double', true, false, false, false);
b  = test_approx_eq(yb, y, name, strpad('BTop expmv equal (GRE)', Ns), 100) && b;
b  = test_approx_eq(ys, y, name, strpad('BTsparse expmv equal (GRE)', Ns), 100) && b;

t  = 0.1*rand();
Ef = expm(t/2*Af);
yb = Ef * conj( Ef * x0(:) );
V  = ExpmvStepper(t, A, [], [], 'prnt', false, 'type', 'SE');
V  = precompute(V, x0);
y  = step(V, x0);
% y  = bt_expmv(t, A, x0, 'prnt', false, 'type', 'SE');
ys = expmv(t/2, As, x0(:), [], 'double', true, false, false, false);
ys = expmv(t/2, As, conj(ys), [], 'double', true, false, false, false);
b  = test_approx_eq(yb, y, name, strpad('BTop expmv equal (SE)', Ns), 100) && b;
b  = test_approx_eq(ys, y, name, strpad('BTsparse expmv equal (SE)', Ns), 100) && b;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix multiplication testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix multiplication
yb = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y  = A*x0;
ys = reshape(As*x0(:), size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse mat*vec', Ns)) && b;
A.Gamma;

% Matrix-transpose multiplication
yb = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize); % symmetric
y  = A.'*x0;
ys = reshape(As.'*x0(:), size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop trans-mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse trans-mat*vec', Ns)) && b;
A.Diag;

% Matrix-conjugate-transpose multiplication
yb = BlochTorreyBrute(x0, conj(Gamma), conj(Dcoeff), Gsize, Vsize);
y  = A'*x0;
ys = reshape(As'*x0(:), size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop conj-trans-mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse conj-trans-mat*vec', Ns)) && b;
A.Gamma;

% Vector*Matrix multiplication (3D grid)
yb = BlochTorreyBrute(conj(x0), Gamma, Dcoeff, Gsize, Vsize);
y  = conj(x0)*A;
ys = reshape(x0(:)'*As, size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop vec-ctrans*mat (3D)', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse vec-ctrans*mat (3D)', Ns)) && b;
A.Diag;

% Vector*Matrix multiplication (1D vector)
yb = BlochTorreyBrute(conj(x0), Gamma, Dcoeff, Gsize, Vsize);
y  = x0(:)'*A;
ys = x0(:)'*As;

b = test_approx_eq(yb, y, name, strpad('BTop vec-ctrans*mat (1D)', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse vec-ctrans*mat (1D)', Ns)) && b;
A.Gamma;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scalar multiplication testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Real-scalar multiplication
a  = randn();
yb = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y  = (a*A)*x0;
ys = reshape((a*As)*x0(:), size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop RHS-real-scalar*mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse RHS-real-scalar*mat*vec', Ns)) && b;
A.Diag;

y = (A*a)*x0;
b = test_approx_eq(yb, y, name, strpad('BTop LHS-real-scalar*mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse LHS-real-scalar*mat*vec', Ns)) && b;
A.Gamma;

% Complex-scalar multiplication
a  = randnc();
yb = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y  = (a*A)*x0;
ys = reshape((a*As)*x0(:), size(x0));

b = test_approx_eq(yb, y, name, strpad('BTop RHS-cplx-scalar*mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse RHS-cplx-scalar*mat*vec', Ns)) && b;
A.Diag;

y = (A*a)*x0;
b = test_approx_eq(yb, y, name, strpad('BTop LHS-cplx-scalar*mat*vec', Ns)) && b;
b = test_approx_eq(yb, ys, name, strpad('BTsparse LHS-cplx-scalar*mat*vec', Ns)) && b;
A.Gamma;

end

function y = BlochTorreyBrute(x0, Gamma, D, Gsize, Vsize)
h = mean(Vsize./Gsize);
y = BlochTorreyAction_brute(x0, h, D, Gamma, 1, false);
end

function A = full_Brute(Gamma, D, Gsize, Vsize)
N = prod(Gsize);
A = zeros(N,N);
ei = zeros(Gsize);

ei(1) = 1;
A(:,1) = vec(BlochTorreyBrute(ei, Gamma, D, Gsize, Vsize));
for ii = 2:size(A,2)
    ei(ii-1) = 0;
    ei(ii) = 1;
    A(:,ii) = vec(BlochTorreyBrute(ei, Gamma, D, Gsize, Vsize));
end

end

function A = full_BTAction(Gamma, D, Gsize, Vsize)
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
tol    = tolfact * eps(maxval);
% tol  = sqrt(tol); % For testing if failures are due to tolerance

maxdiff = max(abs(x(:)-y(:)));
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
