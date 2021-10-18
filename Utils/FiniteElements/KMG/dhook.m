function fd=dhook(p)
x=p(:,1); y=p(:,2);
d1=sqrt(x.*x+y.*y)-1;
d2=sqrt((x+.4).^2+y.^2)-.55;
fd=max([d1 -d2 -y]')';