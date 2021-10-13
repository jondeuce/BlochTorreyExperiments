#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* FMG_LAP_PER_S
 *
 * INPUT ARGUMENTS
 *  x: 3D single array
 *  h: grid spacing (scalar)
 *
 */

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint32_t i, j, k, l;
    
    const float *x = (float*)mxGetData(prhs[0]); //input array
    float h = (nrhs < 2) ? 1.0 : ((float*)mxGetData(prhs[1]))[0]; //grid spacing
    const float h2 = 1.0 / (h * h);
    
    const mwSize *mSize = mxGetDimensions(prhs[0]); //array size
    
    plhs[0] = mxCreateNumericArray(3, mSize, mxSINGLE_CLASS, mxREAL);
    float *dx = (float*)mxGetData(plhs[0]);
    
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NZ     = nxny*(nz-1);
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NX     = nx-1;
    
//     uint32_t xl, xr, yl, yr, zl, zr;
    int32_t yl, yr, zl, zr;
    
    #pragma omp parallel for collapse(2) num_threads(256)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
//             l = k + j;
//             for(i = 0; i < nx; ++i, ++l) {
//                 xl = (i==0 ) ? l+NX : l-1;
//                 xr = (i==NX) ? l-NX : l+1;
//                 yl = (j==0 ) ? l+NY : l-nx;
//                 yr = (j==NY) ? l-NY : l+nx;
//                 zl = (k==0 ) ? l+NZ : l-nxny;
//                 zr = (k==NZ) ? l-NZ : l+nxny;
//                 
//                 dx[l] = -6.0 * x[l] + x[xl] + x[xr] + x[yl] + x[yr] + x[zl] + x[zr];
//             }
            yl = (j==0 ) ?  NY : -nx;
            yr = (j==NY) ? -NY :  nx;
            zl = (k==0 ) ?  NZ : -nxny;
            zr = (k==NZ) ? -NZ :  nxny;
            
            l  = k + j + NX;
            dx[l] = -6.0 * x[l] + x[l-NX] + x[l-1] + x[yl+l] + x[yr+l] + x[zl+l] + x[zr+l];
            
            l -= NX;
            dx[l] = -6.0 * x[l] + x[l+NX] + x[l+1] + x[yl+l] + x[yr+l] + x[zl+l] + x[zr+l];
            
            l++;
            yl += l;
            yr += l;
            zl += l;
            zr += l;
            for(i = 1; i < nx-1; ++i, ++l, ++yl, ++yr, ++zl, ++zr) {
                dx[l] = -6.0 * x[l] + x[l-1] + x[l+1] + x[yl] + x[yr] + x[zl] + x[zr];
            }
        }
    }
    
    if(fabs(h-1.0) > 1e-8) {
        #pragma omp parallel for num_threads(128)
        for(l = 0; l < nxnynz; ++l) {
            dx[l] *= h2;
        }
    }
    
    return;

}

