function [X,mu,Chi2FactorActual] = lsqnonneg_reg(C,d,Chi2Factor)
%
% X = LSQNONNEG_REG(C,d,Chi2Factor) returns the regularized NNLS solution X
% that incurrs an increase in chi^2 by a factor of Chi2Factor.

% Find non-regularized solution
X_noreg=lsqnonneg(C,d);
d_backprojected=C*X_noreg;
residuals=d-d_backprojected;
chi2_min=sum(residuals.^2);
% Initialzation of various components
mu=0;
chi2=chi2_min;
% Minimize energy of spectrum
H=eye(size(C,2));
% Loop to find largest mu that keeps chi-squared in desired range
while chi2(end)<(Chi2Factor*chi2_min)
    % Incrememt mu vector
    if mu(end)>0
        mu=[mu,2*mu(end)];
    else
        mu=[mu,0.001];
    end
    % Compute T2 distribution with smoothing
    smooth=mu(end)*H;
    C_smooth=[C;smooth];
    X_reg=lsqnonneg(C_smooth,[d;zeros(size(C,2),1)]);
    % Find predicted curve and calculate residuals and chi-squared
    d_backprojected=C*X_reg;
    residuals=d-d_backprojected;
    chi2=[chi2,sum(residuals.^2)];
end
% Smooth the chi2(mu) curve using spline fit
% mu_spline=0:0.001:mu(end); %JD
mu_spline=0:0.1e-6:mu(end); %JD
chi2_spline=interp1(mu,chi2,mu_spline,'spline');
% Find the index of the minimum chi-squared satisfying the increase factor
[dummy,min_ind]=min(abs(chi2_spline-(Chi2Factor*chi2_min)));
mu=mu_spline(min_ind);
smooth=mu*H;
% Compute the regularized solution
C_smooth=[C;smooth];
X_reg=lsqnonneg(C_smooth,[d;zeros(size(C,2),1)]);
d_backprojected=C*X_reg;
residuals=d-d_backprojected;
chi2_final=sum(residuals.^2);
% Verify actual chi2 increase factor
Chi2FactorActual=chi2_final/chi2_min;
% Assign output
X=X_reg;

