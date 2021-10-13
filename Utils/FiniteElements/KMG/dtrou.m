function fd=dtrou(p)
r=.25;
xc1=-.5; xc2=.5;  xc3=.5; xc4=-.5;
yc1=-.5; yc2=-.5; yc3=.5; yc4=.5;
dc1=dcircle(p,xc1,yc1,r); dc2=dcircle(p,xc2,yc2,r);
dc3=dcircle(p,xc3,yc3,r); dc4=dcircle(p,xc4,yc4,r);
dc=min(dc1,min(dc2,min(dc3,dc4)));
fd=max(drectangle(p,-1,1,-1,1),-dc);
