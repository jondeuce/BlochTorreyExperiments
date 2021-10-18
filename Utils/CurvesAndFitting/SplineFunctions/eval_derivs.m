function [ cs_df, cs_ddf, df, ddf ] = eval_derivs( cs, t )
%EVAL_DERIVS evaluates the first and second derivatives of the cubic spline
%defined by parameters 'cs' at the points 't'

% extract details from piece-wise polynomial by breaking it apart, and
% make the polynomial that describes the derivative
[breaks,coefs,l,k,d] = unmkpp(cs);
cs_df = mkpp(breaks,repmat(k-1:-1:1,[d*l,1]).*coefs(:,1:k-1),d);

% to calculate 2nd derivative differentiate the 1st derivative
[breaks,coefs,l,k,d] = unmkpp(cs_df);
cs_ddf = mkpp(breaks,repmat(k-1:-1:1,[d*l,1]).*coefs(:,1:k-1),d);

if nargin>1 && nargout>2
    % evaluate the derivatives of the polynomial
    df=ppval(cs_df,t);
    ddf=ppval(cs_ddf,t);
end

end