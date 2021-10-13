#include <stdint.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"


void restrictf(float *x2, uint8_t *M2,
        const float *x, const uint8_t *Mask, const mwSize *siz, const mwSize *siz2);

void restrictd(double *x2, uint8_t *M2,
        const double *x, const uint8_t *Mask, const mwSize *siz, const mwSize *siz2);


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
    const mwSize siz2[3] = {(siz[0]+1) >> 1, (siz[1]+1) >> 1, (siz[2]+1) >> 1};
    
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[1]);
    
    if(is_s) {
        const float *x = (const float*)mxGetData(prhs[0]);
        
        plhs[0] = mxCreateNumericArray(3, siz2, mxSINGLE_CLASS, mxREAL);
        plhs[1] = mxCreateLogicalArray(3, siz2);
        float *x2 = (float*)mxGetData(plhs[0]);
        uint8_t *M2 = (uint8_t*)mxGetData(plhs[1]);
        
        restrictf(x2, M2, x, Mask, siz, siz2);
    } else {
        const double *x = (const double*)mxGetData(prhs[0]);
        
        plhs[0] = mxCreateNumericArray(3, siz2, mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateLogicalArray(3, siz2);
        double *x2 = (double*)mxGetData(plhs[0]);
        uint8_t *M2 = (uint8_t*)mxGetData(plhs[1]);
        
        restrictd(x2, M2, x, Mask, siz, siz2);
    }
    return;
}


void
restrictf(float *x2, uint8_t *M2,
        const float *x, const uint8_t *Mask,
        const mwSize *siz, const mwSize *siz2)
{
    uint32_t i, j, k, l, l2;
    float tmp1, tmp2, tmp4;
    
    const uint32_t nx    = (uint32_t)siz[0];
    const uint32_t ny    = (uint32_t)siz[1];
    const uint32_t nz    = (uint32_t)siz[2];
    const uint32_t nxny  = nx*ny;
    const uint32_t N     = nxny*nz;
    const uint32_t NZ    = nxny*(nz-1);
    const uint32_t NY    = nx*(ny-1);
    const uint32_t NX    = nx-1;
    const uint32_t NZZ   = NZ-nxny;
    const uint32_t NYY   = NY-nx;
    const uint32_t NXX   = NX-1;
    const uint32_t inc_k = 2*nxny;
    const uint32_t inc_j = 2*nx;

    const uint32_t nx2   = (uint32_t)siz2[0];
    const uint32_t ny2   = (uint32_t)siz2[1];
    const uint32_t nz2   = (uint32_t)siz2[2];
    const uint32_t nxny2 = nx2*ny2;
    
    uint32_t ll1[8], ll2[12], ll4[6];
    ll1[0]  = -1-nx-nxny;
    ll1[1]  = -1-nx+nxny;
    ll1[2]  = -1+nx+nxny;
    ll1[3]  = -1+nx-nxny;
    ll1[4]  =  1-nx-nxny;
    ll1[5]  =  1-nx+nxny;
    ll1[6]  =  1+nx+nxny;
    ll1[7]  =  1+nx-nxny;

    ll2[0]  = -1-nxny;
    ll2[1]  = -1+nxny;
    ll2[2]  = -1-nx;
    ll2[3]  = -1+nx;
    ll2[4]  =  1-nxny;
    ll2[5]  =  1+nxny;
    ll2[6]  =  1-nx;
    ll2[7]  =  1+nx;
    ll2[8]  = -nx+nxny;
    ll2[9]  = -nx-nxny;
    ll2[10] =  nx+nxny;
    ll2[11] =  nx-nxny;
    
    ll4[0]  = -1;
    ll4[1]  =  1;
    ll4[2]  = -nx;
    ll4[3]  =  nx;
    ll4[4]  = -nxny;
    ll4[5]  =  nxny;
    
    #pragma omp parallel for private(l, l2, tmp1, tmp2, tmp4) collapse(3)
    for(k = 0; k < N; k += inc_k) {
        for(j = 0; j < nxny; j += inc_j) {
            for(i = 0; i < nx; i += 2) {
                l = k + j + i;
                if(Mask[l]) {
                    l2 = i/2 + nx2*(j/inc_j) + nxny2*(k/inc_k);
                    M2[l2] = 1;
                    
                    if( (i-1 < NXX) && Mask[l+ll4[0]] && Mask[l+ll4[1]] &&
                        (j-nx < NYY) && Mask[l+ll4[2]] && Mask[l+ll4[3]] &&
                        (k-nxny < NZZ) && Mask[l+ll4[4]] && Mask[l+ll4[5]] &&
                        Mask[l+ll2[0]] && Mask[l+ll2[1]] && Mask[l+ll2[2]] &&
                        Mask[l+ll2[3]] && Mask[l+ll2[4]] && Mask[l+ll2[5]] &&
                        Mask[l+ll2[6]] && Mask[l+ll2[7]] && Mask[l+ll2[8]] &&
                        Mask[l+ll2[9]] && Mask[l+ll2[10]] && Mask[l+ll2[11]] &&
                        Mask[l+ll1[0]] && Mask[l+ll1[1]] && Mask[l+ll1[2]] &&
                        Mask[l+ll1[3]] && Mask[l+ll1[4]] && Mask[l+ll1[5]] &&
                        Mask[l+ll1[6]] && Mask[l+ll1[7]]
                      ) {

                        tmp4  = 0.0625f*(
                                x[l+ll4[0]] + x[l+ll4[1]] + x[l+ll4[2]] +
                                x[l+ll4[3]] + x[l+ll4[4]] + x[l+ll4[5]]);
                         
                        tmp2  = 0.03125f*(
                                x[l+ll2[0]] + x[l+ll2[1]] + x[l+ll2[2]] +
                                x[l+ll2[3]] + x[l+ll2[4]] + x[l+ll2[5]] +
                                x[l+ll2[6]] + x[l+ll2[7]] + x[l+ll2[8]] +
                                x[l+ll2[9]] + x[l+ll2[10]] + x[l+ll2[11]]);
                               
                        tmp1  = 0.015625f*(
                                x[l+ll1[0]] + x[l+ll1[1]] + x[l+ll1[2]] +
                                x[l+ll1[3]] + x[l+ll1[4]] + x[l+ll1[5]] +
                                x[l+ll1[6]] + x[l+ll1[7]]);
                        
                        x2[l2] = 0.125f*x[l] + tmp4 + tmp2 + tmp1;
                        
                    } else {
                        x2[l2] = x[l];
                    }
                }
            }
        }
    }
    return;
}


void
restrictd(double *x2, uint8_t *M2,
        const double *x, const uint8_t *Mask,
        const mwSize *siz, const mwSize *siz2)
{
    uint32_t i, j, k, l, l2;
    double tmp1, tmp2, tmp4;
    
    const uint32_t nx    = (uint32_t)siz[0];
    const uint32_t ny    = (uint32_t)siz[1];
    const uint32_t nz    = (uint32_t)siz[2];
    const uint32_t nxny  = nx*ny;
    const uint32_t N     = nxny*nz;
    const uint32_t NZ    = nxny*(nz-1);
    const uint32_t NY    = nx*(ny-1);
    const uint32_t NX    = nx-1;
    const uint32_t NZZ   = NZ-nxny;
    const uint32_t NYY   = NY-nx;
    const uint32_t NXX   = NX-1;
    const uint32_t inc_k = 2*nxny;
    const uint32_t inc_j = 2*nx;

    const uint32_t nx2   = (uint32_t)siz2[0];
    const uint32_t ny2   = (uint32_t)siz2[1];
    const uint32_t nz2   = (uint32_t)siz2[2];
    const uint32_t nxny2 = nx2*ny2;
    
    uint32_t ll1[8], ll2[12], ll4[6];
    ll1[0]  = -1-nx-nxny;
    ll1[1]  = -1-nx+nxny;
    ll1[2]  = -1+nx+nxny;
    ll1[3]  = -1+nx-nxny;
    ll1[4]  =  1-nx-nxny;
    ll1[5]  =  1-nx+nxny;
    ll1[6]  =  1+nx+nxny;
    ll1[7]  =  1+nx-nxny;

    ll2[0]  = -1-nxny;
    ll2[1]  = -1+nxny;
    ll2[2]  = -1-nx;
    ll2[3]  = -1+nx;
    ll2[4]  =  1-nxny;
    ll2[5]  =  1+nxny;
    ll2[6]  =  1-nx;
    ll2[7]  =  1+nx;
    ll2[8]  = -nx+nxny;
    ll2[9]  = -nx-nxny;
    ll2[10] =  nx+nxny;
    ll2[11] =  nx-nxny;
    
    ll4[0]  = -1;
    ll4[1]  =  1;
    ll4[2]  = -nx;
    ll4[3]  =  nx;
    ll4[4]  = -nxny;
    ll4[5]  =  nxny;
    
    #pragma omp parallel for private(l, l2, tmp1, tmp2, tmp4) collapse(3)
    for(k = 0; k < N; k += inc_k) {
        for(j = 0; j < nxny; j += inc_j) {
            for(i = 0; i < nx; i += 2) {
                l = k + j + i;
                if(Mask[l]) {
                    l2 = i/2 + nx2*(j/inc_j) + nxny2*(k/inc_k);
                    M2[l2] = 1;
                    
                    if( (i-1 < NXX) && Mask[l+ll4[0]] && Mask[l+ll4[1]] &&
                        (j-nx < NYY) && Mask[l+ll4[2]] && Mask[l+ll4[3]] &&
                        (k-nxny < NZZ) && Mask[l+ll4[4]] && Mask[l+ll4[5]] &&
                        Mask[l+ll2[0]] && Mask[l+ll2[1]] && Mask[l+ll2[2]] &&
                        Mask[l+ll2[3]] && Mask[l+ll2[4]] && Mask[l+ll2[5]] &&
                        Mask[l+ll2[6]] && Mask[l+ll2[7]] && Mask[l+ll2[8]] &&
                        Mask[l+ll2[9]] && Mask[l+ll2[10]] && Mask[l+ll2[11]] &&
                        Mask[l+ll1[0]] && Mask[l+ll1[1]] && Mask[l+ll1[2]] &&
                        Mask[l+ll1[3]] && Mask[l+ll1[4]] && Mask[l+ll1[5]] &&
                        Mask[l+ll1[6]] && Mask[l+ll1[7]]
                      ) {
                        
                        tmp4  = 0.0625*(
                                x[l+ll4[0]] + x[l+ll4[1]] + x[l+ll4[2]] +
                                x[l+ll4[3]] + x[l+ll4[4]] + x[l+ll4[5]]);
                         
                        tmp2  = 0.03125*(
                                x[l+ll2[0]] + x[l+ll2[1]] + x[l+ll2[2]] +
                                x[l+ll2[3]] + x[l+ll2[4]] + x[l+ll2[5]] +
                                x[l+ll2[6]] + x[l+ll2[7]] + x[l+ll2[8]] +
                                x[l+ll2[9]] + x[l+ll2[10]] + x[l+ll2[11]]);
                               
                        tmp1  = 0.015625*(
                                x[l+ll1[0]] + x[l+ll1[1]] + x[l+ll1[2]] +
                                x[l+ll1[3]] + x[l+ll1[4]] + x[l+ll1[5]] +
                                x[l+ll1[6]] + x[l+ll1[7]]);
                        
                        x2[l2] = 0.125*x[l] + tmp4 + tmp2 + tmp1;
                        
                    } else {
                        x2[l2] = x[l];
                    }
                }
            }
        }
    }
    return;
}
