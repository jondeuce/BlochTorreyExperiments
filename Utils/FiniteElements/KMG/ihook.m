function  [fd,bbox,pfix]=ihook() 
bbox=[-1,0; 1,1]; pfix=[-1,0; 0,1; -0.95,0; 0.15 0]; 
fd=@dhook;  