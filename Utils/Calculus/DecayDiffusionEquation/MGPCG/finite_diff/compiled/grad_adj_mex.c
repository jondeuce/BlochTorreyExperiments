#include <stdint.h>
#include <math.h>
#include <omp.h>
#include "mex.h"


void grad_adjf(float *dx,
        const float *x, const float *y, const float *z,
        const uint8_t *Mask, double *h, const mwSize *siz);

void grad_adjd(double *dx,
        const double *x, const double *y, const double *z,
        const uint8_t *Mask, double *h, const mwSize *siz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 4)
        mexErrMsgTxt("Dx, Dy, Dz, and Mask needed.");
    
    if(!mxIsLogical(prhs[3]))
        mexErrMsgTxt("Mask must be logical.");
    
    const uint8_t is_s = mxIsSingle(prhs[0]);
    const uint8_t is_d = mxIsDouble(prhs[0]);
    
    if(!is_s && !is_d)
        mexErrMsgTxt("Input must be single or double.");
    
    if( (is_s && !mxIsSingle(prhs[1]) && !mxIsSingle(prhs[2])) ||
            (is_d && !mxIsDouble(prhs[1]) && !mxIsDouble(prhs[2])))
        mexErrMsgTxt("Dx, Dy, Dz must have same precision.");
    
    const mwSize *siz = mxGetDimensions(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[3]);
    
    double *h = (double[3]){1.0, 1.0, 1.0};
    if(nrhs > 4) {
        const double *hh = (const double*)mxGetData(prhs[4]);
        h = mxGetNumberOfElements(prhs[4]) < 3
                ? (double[3]){1.0/hh[0], 1.0/hh[0], 1.0/hh[0]}
                : (double[3]){1.0/hh[0], 1.0/hh[1], 1.0/hh[2]};
    }
    
    if(is_s) {
        const float *x = (const float*)mxGetData(prhs[0]);
        const float *y = (const float*)mxGetData(prhs[1]);
        const float *z = (const float*)mxGetData(prhs[2]);
        plhs[0] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        float *dx = (float*)mxGetData(plhs[0]);
        
        grad_adjf(dx, x, y, z, Mask, h, siz);
    } else {
        const double *x = (const double*)mxGetData(prhs[0]);
        const double *y = (const double*)mxGetData(prhs[1]);
        const double *z = (const double*)mxGetData(prhs[2]);
        plhs[0] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        double *dx = (double*)mxGetData(plhs[0]);
        
        grad_adjd(dx, x, y, z, Mask, h, siz);
    }
    return;
}


void 
grad_adjf(float *dx,
        const float *x, const float *y, const float *z,
        const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    
    const float hx = (float)h[0];
    const float hy = (float)h[1];
    const float hz = (float)h[2];
    
    #pragma omp parallel for private(l) collapse(3)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            for(i = 0; i < NX; ++i) {
                l = k + j + i;
                if(Mask[l]) {
                    dx[l+1] += Mask[l+1] ? y[l]*hy : 0.0f;
                    dx[l+nx] += Mask[l+nx] ? x[l]*hx : 0.0f;
                    dx[l+nxny] += Mask[l+nxny] ? z[l]*hz : 0.0f;

                    dx[l] -= (x[l]*hx + y[l]*hy + z[l]*hz);
                }
            }
        }
    }
    return;
}


void 
grad_adjd(double *dx,
        const double *x, const double *y, const double *z,
        const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    
    const double hx = h[0];
    const double hy = h[1];
    const double hz = h[2];
    
    #pragma omp parallel for private(l) collapse(3)
    for(k = 0; k < NZ; k += nxny) {
        for(j = 0; j < NY; j += nx) {
            for(i = 0; i < NX; ++i) {
                l = k + j + i;
                if(Mask[l]) {
                    dx[l+1] += Mask[l+1] ? y[l]*hy : 0.0;
                    dx[l+nx] += Mask[l+nx] ? x[l]*hx : 0.0;
                    dx[l+nxny] += Mask[l+nxny] ? z[l]*hz : 0.0;

                    dx[l] -= (x[l]*hx + y[l]*hy + z[l]*hz);
                }
            }
        }
    }
    return;
}
