#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"


void bgradf(float *dx, float *dy, float *dz,
        const float *x, const uint8_t *Mask, double *h, const mwSize *siz);

void bgradd(double *dx, double *dy, double *dz,
        const double *x, const uint8_t *Mask, double *h, const mwSize *siz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 2)
        mexErrMsgTxt("Phase and Mask needed.");
    
    if(!mxIsLogical(prhs[1]))
        mexErrMsgTxt("Mask must be logical.");
    
    const uint8_t is_s = mxIsSingle(prhs[0]);
    const uint8_t is_d = mxIsDouble(prhs[0]);
    
    if(!is_s && !is_d)
        mexErrMsgTxt("Phase must be single or double.");
    
    const mwSize *siz = mxGetDimensions(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[1]);
    
    double *h = (double[3]){1.0, 1.0, 1.0};
    if(nrhs > 2) {
        const double *hh = (const double*)mxGetData(prhs[2]);
        h = mxGetNumberOfElements(prhs[2]) < 3
                ? (double[3]){1.0/hh[0], 1.0/hh[0], 1.0/hh[0]}
                : (double[3]){1.0/hh[0], 1.0/hh[1], 1.0/hh[2]};
    }
    
    if(is_s) {
        const float *x = (const float*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        float *dx = (float*)mxGetData(plhs[0]);
        float *dy = (float*)mxGetData(plhs[1]);
        float *dz = (float*)mxGetData(plhs[2]);
        
        bgradf(dx, dy, dz, x, Mask, h, siz);
    } else {
        const double *x = (const double*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        double *dx = (double*)mxGetData(plhs[0]);
        double *dy = (double*)mxGetData(plhs[1]);
        double *dz = (double*)mxGetData(plhs[2]);
        
        bgradd(dx, dy, dz, x, Mask, h, siz);
    }
    return;
}


void 
bgradf(float *dx, float *dy, float *dz, 
        const float *x, const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t N    = nxny*nz;
    
    const float hx = (float)h[0];
    const float hy = (float)h[1];
    const float hz = (float)h[2];
    
    #pragma omp parallel for private(l) collapse(3)
    for(k = 0; k < N; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                if(Mask[l]) {
                    dy[l] = i != 0 && Mask[l-1]
                        ? (x[l] - x[l-1])*hy
                        : 0.0f;

                    dx[l] = j != 0 && Mask[l-nx]
                        ? (x[l] - x[l-nx])*hx
                        : 0.0f;

                    dz[l] = k != 0 && Mask[l-nxny]
                        ? (x[l] - x[l-nxny])*hz
                        : 0.0f;
                }
            }
        }
    }
    return;
}


void 
bgradd(double *dx, double *dy, double *dz, 
        const double *x, const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t N    = nxny*nz;
    
    const double hx = h[0];
    const double hy = h[1];
    const double hz = h[2];
    
    #pragma omp parallel for private(l) collapse(3)
    for(k = 0; k < N; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                if(Mask[l]) {
                    dy[l] = i != 0 && Mask[l-1]
                        ? (x[l] - x[l-1])*hy
                        : 0.0;

                    dx[l] = j != 0 && Mask[l-nx]
                        ? (x[l] - x[l-nx])*hx
                        : 0.0;

                    dz[l] = k != 0 && Mask[l-nxny]
                        ? (x[l] - x[l-nxny])*hz
                        : 0.0;
                }
            }
        }
    }
    return;
}
