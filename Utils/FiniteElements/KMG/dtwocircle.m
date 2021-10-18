function d=dtwocircle(p,varargin)
xc1=varargin{1}(1); yc1=varargin{2}(1); rc1=varargin{3}(1);
xc2=varargin{4}(1); yc2=varargin{5}(1); rc2=varargin{6}(1);
d=max(dcircle(p,xc1,yc1,rc1),-dcircle(p,xc2,yc2,rc2));