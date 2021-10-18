function test_conv_diffuse

xb = [0,3000];
x0 = linspacePeriodic(xb(1),xb(2),10);

D  = 3037;
f  = @(x) -complex(R2(x,x0),dW(x,x0));

X  = @(N) linspace(xb(1),xb(2),N).';
U0 = @(N) 1i * ones(N,1);

%--------------------------------------------------------------------------
% Gradient Echo
%--------------------------------------------------------------------------
T  = (10:10:60).'/1000;

N  = 10000;
u1 = [U0(N),parabolicPeriodic1D(D,0,f,U0(N),xb,N,T)];
figure, grid on, hold on;
plot(X(N),abs(u1(:,1:5:end))), title('Exact Solution: Gradient Echo');

N  = 2048;
u2 = [U0(N),simple_convolution(D,0,f,U0(N),xb,N,T)];
% figure, grid on, hold on;
plot(X(N),abs(u2(:,1:5:end))), title('Approximate Solution: Gradient Echo');

S1 = mean(u1,1).';
S2 = mean(u2,1).';

figure, grid on, hold on;
plot([0;T],abs([S1,S2]),'o--'), title('Signal vs. Time: Gradient Echo');

%--------------------------------------------------------------------------
% Spin Echo
%--------------------------------------------------------------------------
dt = 15;
T  = (dt:dt:60).'/1000;
T2 = (dt:dt:30).'/1000;

N  = 50000;
u1 = [U0(N),parabolicPeriodic1D(D,0,f,U0(N),xb,N,T2)];
u1 = [u1,parabolicPeriodic1D(D,0,f,conj(u1(:,end)),xb,N,T2)];
figure, grid on, hold on;
% plot(X(N),abs(u1(:,1:5:end))), title('Exact Solution: Spin Echo');
plot(X(N),abs(u1(:,end))), title('Exact Solution: Spin Echo');

N  = 512;
u2 = [U0(N),simple_convolution(D,0,f,U0(N),xb,N,T2)];
u2 = [u2,simple_convolution(D,0,f,conj(u2(:,end)),xb,N,T2)];
% figure, grid on, hold on;
% plot(X(N),abs(u2(:,1:5:end))), title('Approximate Solution: Spin Echo');
plot(X(N),abs(u2(:,end))), title('Approximate Solution: Spin Echo');

S1 = mean(u1,1).';
S2 = mean(u2,1).';

figure, grid on, hold on;
plot([0;T],abs([S1,S2]),'o--'), title('Signal vs. Time: Spin Echo');

end

function r2 = R2(x,x0)
% r2 = (31.1-14.5)*reshape(any(abs(bsxfun(@minus,x(:),x0))<13.7,2),size(x))+14.5;
r2 = (63.5-14.5)*reshape(any(abs(bsxfun(@minus,x(:),x0))<13.7,2),size(x))+14.5;
end

function dw = dW(x,x0)
y  = bsxfun(@minus,x(:),x0);
b  = abs(y)<13.7;
y  = 270 * sign(y).*(13.7./y).^2;
y(b) = 0;
dw = reshape(sum(y,2),size(x));
end

function u = simple_convolution(D,~,f,u0,xb,N,T)
dt  = mean(diff(T));
vox = diff(xb)/(N-1);

sig = sqrt(2*D*dt);
min_width   =   8; % min width of gaussian (in standard deviations)
width       =	ceil( min_width / (vox/sig) );
unitsum     =   @(x) x/sum(x(:));
Gaussian1   =	unitsum( exp( -0.5 * ( (-width:width).' * (vox/sig) ).^2 ) );
[N1,N2]     =   deal(ceil((N-length(Gaussian1))/2),floor((N-length(Gaussian1))/2));
Kernel      =	[zeros(N1,1); Gaussian1; zeros(N2,1)];
Kernel      =	fft( ifftshift( Kernel ) );

x   = linspace(xb(1),xb(2),N).';
Exp = exp(dt*f(x));

u      = zeros(N,numel(T)+1);
u(:,1) = u0(:);
for ii = 2:size(u,2)
    u(:,ii) = ifftn(fftn(Exp.*u(:,ii-1)).*Kernel);
end
u      = u(:,2:end);
end