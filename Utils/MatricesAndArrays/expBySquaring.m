function [ B ] = expBySquaring( A, n, mult_type )
%EXPBYSQUARING Computes the element-wise matrix power A.^n or the matrix
%power A^n through the method of exponentiating by squaring. See:
%   
%   https://en.wikipedia.org/wiki/Exponentiation_by_squaring
% 
% INPUT ARGUMENTS
%   A:          Matrix to be raised to the integer power n
%   n:          Integer power (may be negative or 0)
%   mult_type:  Type of multiplication:
%                   1 for matrix multiplication
%                   Any other number for element-wise multiplication
% 
% OUTPUT ARGUMENTS
%   B:          A.^n or A^n
% 

if nargin < 3 || isempty(mult_type)
    mult_type	=   0;
end

if n ~= round(n)
    error( 'n must be an integer!' );
end

switch mult_type
    case 1
        B = expBySquaring_Matrixpower(A,n);
    otherwise
        B = expBySquaring_Elementwise(A,n);
end

end

% function y = expBySquaring_Elementwise(x,n)
% 
% if n < 0
%     x = 1 ./ x;
%     n = -n;
% end
% 
% switch n
%     case 1
%         y = x;
%     case 2
%         y = x.*x;
%     case 3
%         y = x.*x.*x;
%     case 4
%         y = x.*x.*x.*x;
%     case 5
%         y = x.*x.*x.*x.*x;
%     case 6
%         y = x.*x.*x.*x.*x.*x;
%     case 7
%         y = x.*x.*x.*x.*x.*x.*x;
%     case 8
%         y = x.*x.*x.*x.*x.*x.*x.*x;
%     otherwise
%         y = ones(size(x),'like',x);
%         while n > 1
%             if mod(n,2) == 0
%                 x = x .* x;
%                 n = n / 2;
%             else
%                 y = x .* y;
%                 x = x .* x;
%                 n = (n - 1) / 2;
%             end
%         end
%         y = x .* y;
% end
% 
% end

function y = expBySquaring_Elementwise(x,n)

if n < 0
    x = 1 ./ x;
    n = -n;
end

switch n
    case 1
        y = x;
    case 2
        y = x.*x;
    case 3
        y = x.*x.*x;
    case 4
        y = x.*x.*x.*x;
    case 5
        y = x.*x.*x.*x.*x;
    case 6
        y = x.*x.*x.*x.*x.*x;
    case 7
        y = x.*x.*x.*x.*x.*x.*x;
    case 8
        y = x.*x.*x.*x.*x.*x.*x.*x;
    otherwise
        %y = ones(size(x),'like',x);
        y = ones(1,'like',x);
        while n > 7
            switch mod(n,8)
                case 0
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = n / 8;
                case 1
                    y = x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 1) / 8;
                case 2
                    y = x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 2) / 8;
                case 3
                    y = x .* x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 3) / 8;
                case 4
                    y = x .* x .* x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 4) / 8;
                case 5
                    y = x .* x .* x .* x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 5) / 8;
                case 6
                    y = x .* x .* x .* x .* x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 6) / 8;
                case 7
                    y = x .* x .* x .* x .* x .* x .* x .* y;
                    x = x .* x .* x .* x .* x .* x .* x .* x;
                    n = (n - 7) / 8;
            end
        end
        switch n
            case 1
                y = x .* y;
            case 2
                y = x .* x .* y;
            case 3
                y = x .* x .* x .* y;
            case 4
                y = x .* x .* x .* x .* y;
            case 5
                y = x .* x .* x .* x .* x .* y;
            case 6
                y = x .* x .* x .* x .* x .* x .* y;
            case 7
                y = x .* x .* x .* x .* x .* x .* x .* y;
        end
end

end

function y = expBySquaring_Matrixpower(x,n)

if n < 0
    x = inv(x);
    n = -n;
end

if n == 1
    y = x;
else
    y = eye(size(x),'like',x);
    while n > 1
        if mod(n,2) == 0
            x = x * x;
            n = n / 2;
        else
            y = x * y;
            x = x * x;
            n = (n - 1) / 2;
        end
    end
    y = x * y;
end

end