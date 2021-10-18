function [v,u,b] = test_fmg_lap()
   
   mSize = [1,1,1];
   dxyz  = 1/256;
   
   f     = @(x,y,z) (sin(5*x*pi) .* sin(5*y*pi) .* sin(5*z*pi));
   truth = @(x,y,z) (f(x,y,z) / (125*pi^2));
   
   nx = 0 : dxyz : 1-dxyz;
   [ix,iy,iz] = ndgrid(nx, nx, nx);
   
   b = double(f(ix, iy, iz));
   u = double(truth(ix, iy, iz));
   
   tic; v = FMG(fmg_lap(u), [], [], @fmg_lap, @fmg_relax_gs_lap); toc;
   
end
