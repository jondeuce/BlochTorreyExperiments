function [Par,Mse] = circleFit(XY)
% Calculates the least-squares fit for a circle given by
%   (x-a)^2+(y-a)^2=r^2
% Par returns [a,b,r]

xx=XY(~isnan(XY(:,1)),1);
yy=XY(~isnan(XY(:,1)),2);

if length(xx)~=length(yy)
    Par=[NaN NaN NaN]; Mse=NaN;
    return;
end

A=[ sum(xx.^2), sum(xx.*yy),sum(xx);
    sum(xx.*yy),sum(yy.^2), sum(yy);
    sum(xx),    sum(yy),    length(xx)  ];

b=[ sum(xx.*(xx.^2+yy.^2));
    sum(yy.*(xx.^2+yy.^2));
    sum(xx.^2+yy.^2)        ];

u=A\b;

Par=[   u(1)/2;
        u(2)/2;
        sqrt(4*u(3)+u(1)^2+u(2)^2)/2;   ];

Mse=rms((sqrt((xx-Par(1)).^2+(yy-Par(2)).^2)-Par(3)));

end