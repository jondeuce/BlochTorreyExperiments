function y = abs2( x )
%ABS2 y = abs2( x ). Returns abs(x).^2 elementwise without needing to take
% the square root of x first.

if isreal(x)
    y = x.*x;
else
    %y = x.*conj(x);
    y = real(x).^2 + imag(x).^2; % avoids cplx mul., and tends to be faster
end

if ~isreal(y)
    y = real(y); % this should never occur, but just in case
end

end

