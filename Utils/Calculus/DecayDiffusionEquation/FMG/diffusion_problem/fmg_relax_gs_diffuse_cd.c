#include <stdint.h>
#include <math.h>
#include "matrix.h"
#include "mex.h"

/* FMG_RELAX_GS_DIFFUSE_CD.c
 *
 * INPUT ARGUMENTS:
 * x:       Initial guess for solution (3D complex double array)
 * c:       modified rhs of problem; c = b/(f+6*D/h^2) (3D complex double array)
 * maxIter: maximum number of iterations (scalar double)
 * g:       modified decay term; g = (D/h^2)/(f+6*D/h^2) (3D complex double array)
 *
 */
void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint32_t maxIter = (uint32_t)mxGetScalar(prhs[2]);
    
//     plhs[0]  = mxDuplicateArray((mxArray*)prhs[0]);
//     plhs[0] = (mxArray*)prhs[0];
//     mxUnshareArray(plhs[0], true);
    
//     mxUnshareArray(prhs[0], true);
//     const double *vr = (const double*)mxGetData(prhs[0]);
//     const double *vi = (const double*)mxGetImagData(prhs[0]);
//     double *vr = (double*)mxGetData(plhs[0]);
//     double *vi = (double*)mxGetImagData(plhs[0]);
    
    const double *cr = (const double*)mxGetData(prhs[1]);
    const double *ci = (const double*)mxGetImagData(prhs[1]);
    const double *gr = (const double*)mxGetData(prhs[3]);
    const double *gi = (const double*)mxGetImagData(prhs[3]);
    
    const mwSize *mSize   = mxGetDimensions(prhs[0]);
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NZ     = nxny*(nz-1);
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NX     = nx-1;
   
    uint32_t i, j, k, l;
    uint32_t il, ir, jl, jr, kl, kr;
    
    /* complex input array */
    const double *xr = (const double*)mxGetData(prhs[0]);
    const double *xi = (const double*)mxGetImagData(prhs[0]);
    
    /* complex output array */
    plhs[0] = mxCreateNumericArray(3, mSize, mxDOUBLE_CLASS, mxCOMPLEX);
    double *vr = (double*)mxGetData(plhs[0]);
    double *vi = (double*)mxGetImagData(plhs[0]);
    
    /* copy initial guess values over */
    #pragma omp parallel for //num_threads(128)
    for(l = 0; l < nxnynz; ++l) {
        vr[l] = xr[l];
        vi[l] = xi[l];
    }     
    
//     mexPrintf("x(12345) = %0.16f + i*%0.16f\n",vr[12344],vi[12344]);
//     mexPrintf("c(12345) = %0.16f + i*%0.16f\n",cr[12344],ci[12344]);
//     mexPrintf("g(12345) = %0.16f + i*%0.16f\n",gr[12344],gi[12344]);
	
    double vsr, vsi;
    while(maxIter--) {
        #pragma omp parallel for collapse(2) //num_threads(128)
        for(k = 0; k < nxnynz; k += nxny) {
            for(j = 0; j < nxny; j += nx) {
                l = k + j;
                for(i = 0; i < nx; ++i, ++l) {
                    il   = (i==0 ) ? l+NX : l-1;
                    ir   = (i==NX) ? l-NX : l+1;
                    jl   = (j==0 ) ? l+NY : l-nx;
                    jr   = (j==NY) ? l-NY : l+nx;
                    kl   = (k==0 ) ? l+NZ : l-nxny;
                    kr   = (k==NZ) ? l-NZ : l+nxny;
                    
                    vsr   = (vr[il] + vr[ir] + vr[jl] + vr[jr] + vr[kl] + vr[kr]);
                    vsi   = (vi[il] + vi[ir] + vi[jl] + vi[jr] + vi[kl] + vi[kr]);
                    vr[l] = gr[l] * vsr - gi[l] * vsi - cr[l];
                    vi[l] = gi[l] * vsr + gr[l] * vsi - ci[l];
                }
            }
        }
    }
    
    return;
    
}
