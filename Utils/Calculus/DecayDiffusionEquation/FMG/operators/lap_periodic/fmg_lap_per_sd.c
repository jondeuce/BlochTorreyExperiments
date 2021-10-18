#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* FMG_LAP_PER_S
 *
 * INPUT ARGUMENTS
 *  x: 3D complex single array
 *  h: grid spacing (scalar)
 *
 */

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint32_t i, j, k, l;
    
    /* grid spacing */
    float h = (nrhs < 2) ? 1.0 : ((float*)mxGetData(prhs[1]))[0];
    const mwSize *mSize = mxGetDimensions(prhs[0]); //array size
    
    /* complex input array */
    float *xr = (float*)mxGetData(prhs[0]);
    float *xi = (float*)mxGetImagData(prhs[0]);
    
    /* complex output array */
    plhs[0] = mxCreateNumericArray(3, mSize, mxSINGLE_CLASS, mxCOMPLEX);
    float *dxr = (float*)mxGetData(plhs[0]);
    float *dxi = (float*)mxGetImagData(plhs[0]);
    
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NZ     = nxny*(nz-1);
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NX     = nx-1;
    
    uint32_t il, ir, jl, jr, kl, kr;
    uint8_t IN_Z0, IN_ZN, IN_Y0, IN_YN, IN_X0, IN_XN;
    
    #pragma omp parallel for collapse(2) //num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = k + j;
            for(i = 0; i < nx; ++i, ++l) {
                il = (i==0 ) ? l+NX : l-1;
                ir = (i==NX) ? l-NX : l+1;
                jl = (j==0 ) ? l+NY : l-nx;
                jr = (j==NY) ? l-NY : l+nx;
                kl = (k==0 ) ? l+NZ : l-nxny;
                kr = (k==NZ) ? l-NZ : l+nxny;
                
                dxr[l] = -6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr];
                dxi[l] = -6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr];
            }
        }
    }
    
    if(fabs(h-1.0) > 1e-8) {
        const float h2 = 1.0 / (h * h);
        #pragma omp parallel for //num_threads(128)
        for(l = 0; l < nxnynz; ++l) {
            dxr[l] *= h2;
            dxi[l] *= h2;
        }
    }
    
    return;

}

