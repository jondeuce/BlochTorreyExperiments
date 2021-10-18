function [ ] = testParabolicPeriodic3D( )
%TESTPARABOLICPERIODIC3D Tests the function parabolicPeriodic1D for various
%cases of transverse magnetization geometries.

%==========================================================================
% Parameters that are kept constant
%==========================================================================
[Nx,Ny,Nz]	=	deal(64);
[xb,yb,zb]	=	deal([-4,4],[-6,6],[-3,3]);
[xc,yc,zc]	=	deal(mean(xb),mean(yb),mean(zb));
[dx,dy,dz]	=   deal(diff(xb),diff(yb),diff(zb));
[hx,hy,hz]	=   deal(dx/Nx,dy/Ny,dz/Nz);

% Peclet Number P = (Decay/Diffusion) = R2max/(Dmax/dx^2)
Tmax	=   0.05;
t       =	linspace(0,Tmax,3).';
x       =	linspace(xb(1),xb(2),Nx+1).';
y       =	linspace(yb(1),yb(2),Ny+1).';
z       =	linspace(zb(1),zb(2),Nz+1).';

% random example
% u0      =   @(x,y,z) complex( 0, 1 + 0.2 * bsxfun3D(@plus, ((2/dx)*(x-xc)).^2, ((2/dy)*(y-yc)).^2, ((2/dz)*(z-zc)).^2 ) );
u0      =   @(x,y,z) complex( 0, bsxfun3D(@plus, ones(size(x)), ones(size(y)), ones(size(z))) );

% exact solution
% [a,b,c,alpha]	=   deal(1,1,1,3+rand);
% u0      =	@(x,y,z) times3D(sin(a*single(x)),sin(b*single(y)),sin(c*single(z)));
% f       =	@(x,y,z) ((1/3)*(alpha-a^2-b^2-c^2))*plus3D(ones(size(x),'single'),ones(size(y),'single'),ones(size(z),'single'));
% u0      =	@(x,y,z) 1+plus3D(a*single(x.*x),b*single(y.*y),c*single(z.*z));
% f       =	@(x,y,z) alpha + (2.*(a+b+c))./u0(x,y,z);
% u0      =	@(x,y,z) exp(plus3D((-0.5.*a).*(x-xc).^2,(-0.5.*b).*(y-yc).^2,(-0.5.*c).*(z-zc).^2));
% f       =	@(x,y,z) (alpha-a-b-c) + plus3D(a^2.*(x-xc).^2,b^2.*(y-yc).^2,c^2.*(z-zc).^2);

% fu      =	@(x,y,z,t) bsxfun(@times,u0(x,y,z),reshape(exp(-alpha*t(:)),1,1,1,[]));
% Iu0     =   integral3(u0,xb(1),xb(2),yb(1),yb(2),zb(1),zb(2),'abstol',1e-5,'reltol',1e-3);
% fS      =   @(t) exp(-alpha*t)*Iu0;

s       =   0.1; %stdev
Line	=	@(x,xb,yb) diff(yb)/diff(xb)*(x-xb(1)) + yb(1);
Box     =	@(x)	(abs(x)<1) + 0.5*(abs(x)==1);
SoftBox	=	@(x)	0.5 * (erf((x+1)./(sqrt(2)*s))-erf((x-1)./(sqrt(2)*s)));
BoxFun	=   SoftBox;

% Syntax: y = HumpFun( BoxFun, centers, widths, heights, x )
dwx	=   @(x,dwamp)	HumpFun( BoxFun, linspacePeriodic(xb(1),xb(2),3), dx/12, dwamp*[1,1,-1], x );
dwy	=   @(y,dwamp)	HumpFun( BoxFun, linspacePeriodic(yb(1),yb(2),4), dy/16, dwamp*[-1,1,-1,1], y );
dwz	=   @(z,dwamp)	HumpFun( BoxFun, linspacePeriodic(zb(1),zb(2),5), dz/20, dwamp*[-1,1,1,1,-1], z );
dw	=   @(x,y,z,th)	bsxfun3D(@plus,dwx(x,th/Tmax),dwy(y,th/Tmax),dwz(z,th/Tmax));

r2x	=   @(x,R2amp) HumpFun( BoxFun, linspacePeriodic(xb(1),xb(2),5), dx/20, R2amp*[1,1,-1,1,-1], x );
r2y	=   @(y,R2amp) HumpFun( BoxFun, linspacePeriodic(yb(1),yb(2),3), dy/12, R2amp*[1,-1,1], y );
r2z	=   @(z,R2amp) HumpFun( BoxFun, linspacePeriodic(zb(1),zb(2),2), dz/8, R2amp*[1,-1], z );
R2	=   @(x,y,z,R2dc,R2amp)	R2dc + bsxfun3D(@plus,r2x(x,R2amp),r2y(y,R2amp),r2z(z,R2amp));

%==========================================================================
% Vary diffusion coefficient and dephasing; R2-decay is constant
%==========================================================================
D_Small     =   0.2;
D_Large     =   1.0;
th_Small	=   pi/6;
th_Large	=   3*pi/4;
R2dc        =   1;

D_vals      =   [D_Large];
th_vals     =   [th_Large];
R2amp_vals	=   [0.7*R2dc];

% D_labels	=   { 'Small Diffusion', 'Large Diffusion' };
% th_labels	=   { 'No Dephasing', 'Large Dephasing' };
% R2_labels	=   { 'Uniform R2-Decay', 'Non-Uniform R2-Decay' };
D_labels	=   { 'Large Diffusion' };
th_labels	=   { 'Large Dephasing' };
R2_labels	=   { 'Non-Uniform R2-Decay' };

close all force

for ii = 1:numel(th_vals)
    for jj = 1:numel(D_vals)
        for kk = 1:numel(R2amp_vals)
            R2amp	=   R2amp_vals(kk);
            D       =   D_vals(jj);
            th      =   th_vals(ii);
            labs	=   [R2_labels{kk} ', ' D_labels{jj} ', ' th_labels{ii}];
%             labs	=   '';
            
            [a,b,c]	=   deal(D);
            f       =   @(x,y,z) complex( R2(x,y,z,R2dc,R2amp), dw(x,y,z,th) );
            d       =	@(x,y,z) -f(x,y,z);
            
            % Solve with parabolicPeriodic1D
            [~,u,~]	=	parabolicPeriodic3D( a, b, c, d, u0, xb, yb, zb, Nx+1, Ny+1, Nz+1, t, 'single', [] );
            S       =	squeeze( sum(sum(sum(u,1),2),3) );
            plotMagnetization3D(x,y,z,t,xb,yb,zb,u,S,labs);
            
            % Solve via spectral 'closed' form
            [fSe,fue,~,~,~]	=	heatLikeEqn3D(...
                D, D, D, f, u0, 4, xb, yb, zb, 4*Nx, 4*Ny, 4*Nz, 'cubic' );
            Se	=   fSe(t);
            ue	=   fue(x,y,z,t);
            plotMagnetization3D(x,y,z,t,xb,yb,zb,ue,Se,['Closed Form: ' labs]);
            
            % Exact solution
%             FU	=   fu(x,y,z,t);
%             FS	=   fS(t);
%             
%             % Plot error
%             plotMagnetization3D(x,y,z,t,xb,yb,zb,abs(FU-ue),abs(FS-Se),['Error: ue, Se' labs]);
%             plotMagnetization3D(x,y,z,t,xb,yb,zb,abs(FU-u), abs(FS-S), ['Error: u, S' labs]);
            
            % Plot error
            plotMagnetization3D(x,y,z,t,xb,yb,zb,abs(u-ue),abs(S-Se),['Error: ' labs]);
            
            %plot_u_error(u,x,a,b,c,xb,N,t,labs);
            %plot_ue_error(ue,x,t,a,b,c,labs);
            %plot_u_error(Ue,x,a,b,c,xb,N,t,['Closed Form: ' labs]);
            
            drawnow
        end
    end
end

end

function [ue,Se] = get_closedForms_NoDiffusion(R,u0,xb,N,T)

x	=   linspace(xb(1),xb(2),N);
ue	=   zeros(N,numel(T));
u0x	=   u0(x);
Rx	=   R(x);
for jj = 1:numel(T)
    t	=   T(jj);
    ue(:,jj)	=   u0x .* exp( -Rx * t );
end
Se	=   abs( [sum(u0x), sum(ue,1)] );

end

function y = HumpFun( BoxFun, centers, widths, heights, x )

len	=   max( [numel(centers), numel(widths), numel(heights)] );
e	=   ones(len,1,'like',x);
[ centers, widths, heights ]	=   ...
    deal( e.*centers(:), e.*widths(:), e.*heights(:) );

y	=   heights(1) * BoxFun( (2.0/widths(1)) * (x-centers(1)) );
for ii = 2:length(centers)
    y	=	y + heights(ii) * BoxFun( (2.0/widths(ii)) * (x-centers(ii)) );
end

end

function [u,S] = get_S_closedForm(D,f,u0,Tmax)

[S,u]	=	heatLikeEqn1D(D,f,u0,Tmax);
iS0     =   1/abs(S(0));
S       =	@(t) iS0 * abs(S(t)); %normalized signal

end

function err = check_Lu(u,jj,a,b,c,xb,N,T)

ii	=   2:N-1;
x	=   linspace(xb(1),xb(2),N).';
xii	=   x(ii);

dt	=   T(2)-T(1); % assume constant
dx	=   x(2)-x(1); % assume constant

U	=   u(ii,jj);
Ut	=   (u(ii,jj+1)-u(ii,jj-1))/(2*dt);
Ux	=   (u(ii+1,jj)-u(ii-1,jj))/(2*dx);
Uxx	=   (u(ii+1,jj)-2*U+u(ii-1,jj))/dx^2;

err	=   abs( Ut - ( bsxfun(@times,a(xii),Uxx) + ...
                    bsxfun(@times,b(xii),Ux) + ...
                    bsxfun(@times,c(xii),U) ) );

end

function err = check_Lue(ue,x,t,a,b,c)

dt	=   1e-3; % assume constant
dx	=   1e-3; % assume constant

x	=   x(:);
t	=   t(:);

U	=   ue(x,t);
Ut	=   (ue(x,t+dt)-ue(x,t-dt))/(2*dt);
Ux	=   (ue(x+dx,t)-ue(x-dx,t))/(2*dx);
Uxx	=   (ue(x+dx,t)-2*U+ue(x-dx,t))/(dx^2);

err	=   abs( Ut - ( bsxfun(@times, a(x), Uxx) + ...
                    bsxfun(@times, b(x), Ux) + ...
                    bsxfun(@times, c(x), U ) ) );

end

function plot_u_error(u,x,a,b,c,xb,N,t,titl)

if nargin < 9; titl = ''; end

err     =	check_Lu( u, 2:size(u,2)-1, @(x)a, @(x)b, @(x) c(x), xb, N, t );
[xx,tt] =	meshgrid( x(2:end-1), t(2:end-1) );

figure, hold on
surf(xx,tt,err')
xlabel('x'); xlim([min(x),max(x)]);
ylabel('t'); ylim([0,t(end)]);
zlabel('error'); zlim([0,0.1]);
title(['Numerical Solution Error: ', titl]);
view([0,-1,0]);

end

function plot_ue_error(ue,x,t,a,b,c,titl)

if nargin < 7; titl = ''; end

err     =	check_Lue(ue,x,t,@(x)a,@(x)b,@(x)c(x));
[xx,tt] =	meshgrid( x, t );

figure, hold on
surf(xx,tt,err.')
xlabel('x'); xlim([min(x),max(x)]);
ylabel('t'); ylim([0,t(end)]);
zlabel('error'); zlim([0,0.1]);
title(['Analytical Solution Error: ', titl]);
view([0,-1,0]);

end

function plotMagnetization3D(x,y,z,t,xb,yb,zb,u,S,titl)

if length(t) >= 3
    tidx	=   round( linspace(1,length(t),3) );
else
    tidx	=   1:length(t);
end

if isa( u, 'function_handle' )
    u	=	u(x,y,z,t(tidx));
else
    u	=   u(:,:,:,tidx);
end

if isa( S, 'function_handle' )
    tt	=   linspace(t(1),t(end),20).';
    S	=	S(tt);
else
    tt	=   t(:);
    S	=   S(:);
end
Su      =   abs( squeeze(sum(sum(sum(u,1),2),3)) );

slices	=   [ randi(size(u,1)), randi(size(u,2)), randi(size(u,3)) ];

[Y_yz,Z_yz]	=   meshgrid(y,z);
[X_xz,Z_xz]	=   meshgrid(x,z);
[X_xy,Y_xy]	=   meshgrid(x,y);

figure,
Umax_X	=   max(abs(reshape(u(slices(1),:,:,:),[],1)));
Umax_Y	=   max(abs(reshape(u(:,slices(2),:,:),[],1)));
Umax_Z	=   max(abs(reshape(u(:,:,slices(3),:),[],1)));
Nt      =   length(tidx);
for ii = 1:Nt
    
    subplot(Nt,3,1+3*(ii-1)), hold on
    U	=   squeeze(u(slices(1),:,:,ii));
    surf(Y_yz,Z_yz,abs(U));
    xlim(yb);
    ylim(zb);
    zlim([0,Umax_X]);
    view([1,1,0.5]);
    if ii == 1, title(sprintf('X-slice: %d/%d %s', slices(1), size(u,1), titl)); end
    
    subplot(Nt,3,2+3*(ii-1)), hold on
    U	=   squeeze(u(:,slices(2),:,ii));
    surf(X_xz,Z_xz,abs(U));
    xlim(xb);
    ylim(zb);
    zlim([0,Umax_Y]);
    view([1,1,0.5]);
    if ii == 1, title(sprintf('Y-slice: %d/%d %s', slices(2), size(u,2), titl)); end
    
    subplot(Nt,3,3+3*(ii-1)), hold on
    U	=   u(:,:,slices(3),ii);
    surf(X_xy,Y_xy,abs(U));
    xlim(xb);
    ylim(yb);
    zlim([0,Umax_Z]);
    view([1,1,0.5]);
    if ii == 1, title(sprintf('Z-slice: %d/%d %s', slices(3), size(u,3), titl)); end
    
end

end