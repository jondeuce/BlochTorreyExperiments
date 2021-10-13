function [fd,bbox,pfix]=idcavity()
fd=@ddcavity;
bbox=[-.25,-1;1.25,0.25];
pfix=[-.25,0;0,0;0,-1;1,-1;1,0;1.25,0;1.25,0.25;-0.25,0.25];