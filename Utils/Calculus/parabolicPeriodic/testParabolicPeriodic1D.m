function [ ] = testParabolicPeriodic1D( )
%TESTPARABOLICPERIODIC1D Tests the function parabolicPeriodic1D for various
%cases of transverse magnetization geometries.

%==========================================================================
% Parameters that are kept constant
%==========================================================================
xb	=	[-pi,pi];
dxb	=   diff(xb);
N	=	150;
Tmax=   2.0;
t	=	linspace(0,Tmax,100);
T	=	t(2:end);
x	=	linspace(xb(1),xb(2),N).';
%u0	=	@(x) 1i*(1+0.3*sin(x));
u0	=   @(x) 1i*(1+0.2*(x/pi).^2);

Line	=	@(x,xb,yb) diff(yb)/diff(xb)*(x-xb(1)) + yb(1);
Box     =	@(x,s)	(abs(x)<1) + 0.5*(abs(x)==1);
SoftBox	=	@(x,s)	0.5 * (erf((x+1)./(sqrt(2)*s))-erf((x-1)./(sqrt(2)*s)));
BoxFun	=   SoftBox;

s	=   0.1;
dw	=   @(x,th)	(th/max(T)) * BoxFun( (x-(xb(1)+0.25*dxb))/(0.25*dxb/2), s ) - ...
                (th/max(T)) * BoxFun( (x-(xb(1)+0.75*dxb))/(0.25*dxb/2), s );
R2	=   @(x,R2dc,R2amp)	...
    R2dc -  R2amp * BoxFun( (x-(xb(1)+0.25*dxb))/(0.10*dxb/2), s ) ...
         +	R2amp * BoxFun( (x-(xb(1)+0.60*dxb))/(0.30*dxb/2), s );

D_Small	=   0.2;
D_Large	=   2.0;
th_Small=   pi/6;
th_Large=   3*pi/4;
R2dc	=   1;

%==========================================================================
% Vary diffusion coefficient and dephasing; R2-decay is constant
%==========================================================================
D_vals      =   [D_Large];
th_vals     =   [th_Large];
R2amp_vals	=   [R2dc/1.5];

% D_labels	=   { 'Small Diffusion', 'Large Diffusion' };
% th_labels	=   { 'No Dephasing', 'Large Dephasing' };
% R2_labels	=   { 'Uniform R2-Decay', 'Non-Uniform R2-Decay' };
D_labels	=   { 'Large Diffusion' };
th_labels	=   { 'Large Dephasing' };
R2_labels	=   { 'Non-Uniform R2-Decay' };

for ii = 1:numel(th_vals)
    for jj = 1:numel(D_vals)
        for kk = 1:numel(R2amp_vals)
            R2amp	=   R2amp_vals(kk);
            D       =   D_vals(jj);
            th      =   th_vals(ii);
            labs	=   [R2_labels{kk} ', ' D_labels{jj} ', ' th_labels{ii}];
            
            a       =	D;
            b       =	0;
            f       =   @(x) complex( R2(x,R2dc,R2amp), dw(x,th) );
            c       =	@(x) -f(x);
            
            % Solve with parabolicPeriodic1D
            %u	=	parabolicPeriodicDeriv1D(a,b,c,u0,xb,N,t);
            u	=	parabolicPeriodic1D(a,b,c,u0,xb,N,t);
            S	=	abs( sum(u,1) );
            plotMagnetization1D(x,t,xb,u,S,labs);
            
            % Solve via spectral 'closed' form
            [ue,Se]	=	get_S_closedForm(D,f,u0,Tmax);
            Ue	=   ue(x,t);
            Se	=   abs(Se(t));
            plotMagnetization1D(x,t,xb,Ue,Se,['Closed Form: ' labs]);
            
            tt      =   linspace(0,t(end),100);
            Se_ue	=   abs( sum( ue(x,tt), 1 ) );
            subplot(1,2,2);
            plot( tt, Se_ue/Se_ue(1), 'r--', 'linewidth', 3 );
            
            %plot_u_error(u,x,a,b,c,xb,N,t,labs);
            %plot_ue_error(ue,x,t,a,b,c,labs);
            %plot_u_error(Ue,x,a,b,c,xb,N,t,['Closed Form: ' labs]);
            
            %if ~isequal([ii,jj,kk],[1,1,1]), input('paused'); end
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

function [ue,Se] = get_closedForms_cn(a,b,c,u0,xb,N,T)
% CURRENTLY DOESN'T WORK

a	=   @(x) a.*ones(size(x)); % constant diffusion coeff
b	=   @(x) b.*zeros(size(x)); % constant zero

ue	=	cn1D( a, b, c, u0, xb, N-1, T(end), length(T) );
Se	=   abs( sum(ue,1) );
ue	=   ue(:,2:end);

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

function plotMagnetization1D(x,t,xb,u,S,titl)

figure, hold on

subplot(1,2,1), hold on
u	=   abs(u);
if size(u,2) >= 5
    idx	=   round( linspace(1,size(u,2),5) );
    u	=   u(:,idx);
end

plot(x,u,'b-','linewidth',4);
xlim(xb);
ylim([0,max(u(:))]);
title(titl);

subplot(1,2,2), hold on
plot(t,S/S(1),'g-','linewidth',4,'marker','.','markeredgecolor','b','markersize',30);
xlim([0,t(end)]);
ylim([0,1]);
title(titl);

end

% Substitution (maybe not useful):
%	u(x,t)	=	exp(-f(x)*t) * v(x,t)
%          :=	E*v
% 
%   u_t     =	D*u_xx - f*u
%   u_t     =   E_t*v + E*v_t
%           =  -f*E*v + E*v_t
%   u_x     =   E_x*v + E*v_x
%   u_xx	=   E_xx*v + 2*E_x*v_x + E*v_xx
%           =   (t^2*f'(x)^2 - t*f''(x))*E*v + 2*(-t*f'(x))*E*v_x + E*v_xx
% 
% E * (-f*v + v_t)	=
%   E * [ D * [(t^2*f'^2 - t*f'')*v + 2*(-t*f')*v_x + v_xx] - f*v ]
% 
%	v_t     =   D * [v_xx + (-2*t*f')*v_x + (t^2*f'^2 - t*f'')*v]
