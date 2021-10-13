function v = atan2_2pi(y,x)
%ATAN2_2PI this function returns the arctan of y/x in the range [0,2*pi)

v = atan2(y,x);
v = v + 2*pi*(v<0);

end