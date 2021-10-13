function d=ddcavity(p)
d1=drectangle(p,0,1,-1,0);
d2=drectangle(p,-.25,1.25,0,0.25);
d=min(d1,d2);