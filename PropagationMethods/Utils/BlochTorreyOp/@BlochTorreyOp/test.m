function test

rng('default')

Gsize = [8,11,6];
Vsize = 1000*rand()*(Gsize./max(Gsize));
h = mean(Vsize./Gsize);
x0 = randnc(Gsize);

% Gamma randnc, Dcoeff scalar positive
Gamma = randnc(Gsize);
Dcoeff = rand()*h^2;
run_suite_combinations('Gamma randnc, Dcoeff scalar positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma randnc, Dcoeff scalar negative
Gamma = randnc(Gsize);
Dcoeff = -rand()*h^2;
run_suite_combinations('Gamma randnc, Dcoeff scalar negative', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma randnc, Dcoeff randnc
Gamma = randnc(Gsize);
Dcoeff = randnc()*h^2;
run_suite_combinations('Gamma randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma scalar randnc, Dcoeff randnc
Gamma = randnc();
Dcoeff = randnc()*h^2;
run_suite_combinations('Gamma scalar randnc, Dcoeff randnc', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma scalar randnc, Dcoeff positive
Gamma = randnc();
Dcoeff = rand()*h^2;
run_suite_combinations('Gamma scalar randnc, Dcoeff positive', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma == -1, Dcoeff == 0
Gamma = -1;
Dcoeff = 0;
run_suite_combinations('Gamma == -1, Dcoeff == 0', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma zeros, Dcoeff const array
Gamma = zeros(Gsize);
Dcoeff = rand()*ones(Gsize);
run_suite_combinations('Gamma zeros, Dcoeff const array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma zeros, Dcoeff randn array
Gamma = zeros(Gsize);
Dcoeff = randn(Gsize);
run_suite_combinations('Gamma zeros, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

% Gamma randnc, Dcoeff randn array
Gamma = randnc(Gsize);
Dcoeff = randn(Gsize);
run_suite_combinations('Gamma randnc, Dcoeff randn array', ...
    x0, Gamma, Dcoeff, Gsize, Vsize);

end

function run_suite_combinations(name, x0, Gamma, Dcoeff, Gsize, Vsize)

% run all (necessary) combinations of real/complex inputs
[xreal, greal, dreal] = deal(isreal(x0), isreal(Gamma), isreal(Dcoeff));
[c1,c2,c3,c4,c5,c6,c7,c8] = deal(true, ...
    ~xreal, ~greal, ~dreal, ...
    ~xreal && ~greal, ~xreal && ~dreal, ~greal && ~dreal, ...
    ~xreal && ~greal && ~dreal);

c1 && run_suite(name, x0,       Gamma,       Dcoeff,       Gsize, Vsize);
c2 && run_suite(name, real(x0), Gamma,       Dcoeff,       Gsize, Vsize);
c3 && run_suite(name, x0,       real(Gamma), Dcoeff,       Gsize, Vsize);
c4 && run_suite(name, x0,       Gamma,       real(Dcoeff), Gsize, Vsize);
c5 && run_suite(name, real(x0), real(Gamma), Dcoeff,       Gsize, Vsize);
c6 && run_suite(name, real(x0), Gamma,       real(Dcoeff), Gsize, Vsize);
c7 && run_suite(name, x0,       real(Gamma), real(Dcoeff), Gsize, Vsize);
c8 && run_suite(name, real(x0), real(Gamma), real(Dcoeff), Gsize, Vsize);

fprintf('\n');

end

function b = run_suite(name, x0, Gamma, Dcoeff, Gsize, Vsize)

Ns = 35; % string message pad length

A = BlochTorreyOp(Gamma, Dcoeff, Gsize, Vsize);
As = sparse(A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix multiplication testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix multiplication
y_BTbrute = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y_BTop = A*x0;
y_BTsparse = reshape(As*x0(:), size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse mat*vec',Ns));

% Matrix-transpose multiplication
y_BTbrute = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize); % symmetric
y_BTop = A.'*x0;
y_BTsparse = reshape(As.'*x0(:), size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop trans-mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse trans-mat*vec',Ns));

% Matrix-conjugate-transpose multiplication
y_BTbrute = BlochTorreyBrute(x0, conj(Gamma), conj(Dcoeff), Gsize, Vsize);
y_BTop = A'*x0;
y_BTsparse = reshape(As'*x0(:), size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop conj-trans-mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse conj-trans-mat*vec',Ns));

% Vector*Matrix multiplication
y_BTbrute = BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y_BTop = x0*A;
y_BTsparse = reshape(x0(:).'*As, size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop vec*mat',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse vec*mat',Ns));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scalar multiplication testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Real-scalar multiplication
a = randn();
y_BTbrute = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y_BTop = (a*A)*x0;
y_BTsparse = reshape((a*As)*x0(:), size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop RHS-real-scalar*mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse RHS-real-scalar*mat*vec',Ns));

y_BTop = (A*a)*x0;
test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop LHS-real-scalar*mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse LHS-real-scalar*mat*vec',Ns));

% Complex-scalar multiplication
a = randnc();
y_BTbrute = a*BlochTorreyBrute(x0, Gamma, Dcoeff, Gsize, Vsize);
y_BTop = (a*A)*x0;
y_BTsparse = reshape((a*As)*x0(:), size(x0));

test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop RHS-cplx-scalar*mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse RHS-cplx-scalar*mat*vec',Ns));

y_BTop = (A*a)*x0;
test_approx_eq(y_BTbrute, y_BTop, name, strpad('BTop LHS-cplx-scalar*mat*vec',Ns));
test_approx_eq(y_BTbrute, y_BTsparse, name, strpad('BTsparse LHS-cplx-scalar*mat*vec',Ns));

% Success
b = true;

end

function y = BlochTorreyBrute(x0, Gamma, D, Gsize, Vsize)
h = mean(Vsize./Gsize);
y = BlochTorreyAction_brute(x0, h, D, Gamma, 1, false);
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

function b = test_approx_eq(x,y,name,msg,tol)
if nargin < 5; tol = 10*max(eps(max(abs(x(:)))), eps(max(abs(y(:))))); end
if nargin < 4; msg = 'test failed'; end
if nargin < 3; name = 'N/A'; end
% tol = sqrt(tol);
maxdiff = max(abs(x(:)-y(:)));
b = (maxdiff <= tol);
if ~b; warning(errmsg(name,msg)); else; fprintf('%s\n',passedmsg(name,msg)); end
end
