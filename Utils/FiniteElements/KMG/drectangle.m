function d=drectangle(p,x1,x2,y1,y2)

%   Copyright (C) 2004 Per-Olof Persson. See COPYRIGHT.TXT for more details.

d=-min(min(min(-y1+p(:,2),y2-p(:,2)),-x1+p(:,1)),x2-p(:,1));
