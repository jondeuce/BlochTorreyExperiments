%Created:       05.07.2012 by Peter Kandolf
%Last edit: 	06.07.2012 by Peter Kandolf
%Version:       0.1
%Author:        Marco Caliari
%
%Remarks:
%   - If the matrix is sparse the original version is used, otherwise a low
%   storage version that is faster for full matrices. 
%
%Interface:
% EXTREIGS = GERSH (A,VARARGIN)
%
% The matrix is split into the symmetric and the skew-symmetric part
% LR, SR (real part), LI (imaginary part). By specifying any additional
% parameter (e.g. []) a variant with loop is computed (ATTENTION slow for
% sparse matrices but decreased storage demands).
%
% EXTREIGS=struct('SR', SR, 'LR', LR, 'LI2', LI2)
%
%See also PHILEJA
%--------------------------------------------------------------------------
%Changes:
%   07.11.14 (PK):  fix for complex full matrix where SI was computed wrong
%   05.07.12 (PK):  changes in version 0.1
%                   file created
%                   introduced new alternative computation as for the
%                   orginal version the whole matrix was copied to avoid a
%                   loop but actually this results in a massive
%                   storage overhead for full matrices and has no
%                   segnificant speed benefits. It actually is slower (for
%                   full matrices). Also changed the order of abs and diag
%                   in the original code and deleted unecessary zeros in 
%                   the computation of LI to increase speed. The problem is
%                   that for sparse matrices the row selection is very
%                   slow and therefore the original version should not be
%                   replaced at any time.
function extreigs = gersh (A,varargin)
if issparse(A) && nargin==1
    %Original version
    AS = (A + A') / 2;
    radius = sum (abs (AS), 1)';
    extreigs.SR = full (min (diag (AS) - (radius - abs (diag (AS)))));
    extreigs.LR = full(max(diag (AS) + (radius - abs(diag(AS)))));
    AS = (A - AS);
    radius = sum (abs(AS),1)';
    extreigs.SI = full (min ((imag(diag (AS))) - (radius - abs (diag (AS)))));
    extreigs.LI = full(max(imag(diag (AS)) + (radius - abs(diag(AS)))));
else
    radius=zeros(size(A,2),2);
    for i=1:size(A,2)
        radius(i,1)=sum(abs(A(i,:)'+A(:,i)))/2;
        radius(i,2)=sum(abs(A(i,:)'-A(:,i)))/2;
    end
    d=real(diag(A));
    extreigs.SR=(min(d - (radius(:,1) - abs(d))));
    extreigs.LR=(max(d + (radius(:,1) - abs(d))));

    d=imag(diag(A));
    extreigs.SI = min(d - (radius(:,2)-abs(d)));
    extreigs.LI = max(d + (radius(:,2)-abs(d)));
    extreigs.LI2=(max(radius(:,2)))^2;
end