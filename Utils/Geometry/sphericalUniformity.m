function [ c_lm, L, M ] = sphericalUniformity( x, y, z, lmax, plot_coeffs )
%SPHERICALUNIFORMITY Summary of this function goes here
%   Detailed explanation goes here

if nargin < 5; plot_coeffs = false; end
if nargin < 4; lmax = 5; end

% Note: spherefun uses the convention Y_lm = Y_lm(lambda, theta) where
% lambda is the azimuthal angle and theta the polar angle, and Y_lm are the
% real spherical harmonics (as opposed to the complex spherical harmonics
% as used e.g. in J.D. Jackson Classical Electrodynamics).
[ th, lambda, r ] = cart2sph_physics( x, y, z );

k = 1;
[c_lm, L, M] = deal(zeros((lmax+1)^2, 1));

for l = 0:lmax
    for m = -l:l
        Y = spherefun.sphharm(l,m);
        
        % For continuous function f, you project f onto Y
        %q_lm(k,1) = sum2(f.*Y); 
        
        % For discrete points, we take f to be a sum of delta functions,
        % and integrate by directly evaluating Y at locations phi, th, r
        % 
        % See equations (1) and (2) in:
        %    http://www.chebfun.org/examples/sphere/SphericalHarmonics.html
        %
        % The term r.^l below comes from the analogy with multipole moments
        % of electrostatics; when [x,y,z] are on the sphere and r=1 there
        % is no difference. See e.g. equation (4.3) in J.D. Jackson, 3e.
        c_lm(k) = sum( Y(lambda, th) .* r.^l );
        L(k) = l;
        M(k) = m;
        
        k = k + 1;
    end
end

if plot_coeffs
    stem3(L, M, abs(c_lm), 'filled')
    
    ylim([-lmax, lmax])
    set(gca,'ZScale','log')
    set(gca,'Xdir','reverse')
    view([-13 18])
    xlabel('$\ell$','Interpreter','Latex')
    ylabel('m')
    zlabel('$|q_{lm}|$')
end

end

