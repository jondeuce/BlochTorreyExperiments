#include <stdint.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"


void prolongf(float *x, const uint8_t *Mask,
        const float *x2, const uint8_t *M2,
        const mwSize *siz, const mwSize *siz2);

void prolongd(double *x, const uint8_t *Mask,
        const double *x2, const uint8_t *M2,
        const mwSize *siz, const mwSize *siz2);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 3)
        mexErrMsgTxt("Phase and two Masks needed.");
    
    if(!mxIsLogical(prhs[1]) && !mxIsLogical(prhs[2]))
        mexErrMsgTxt("Both Masks must be logical.");
    
    const uint8_t is_s = mxIsSingle(prhs[0]);
    const uint8_t is_d = mxIsDouble(prhs[0]);
    
    if(!is_s && !is_d)
        mexErrMsgTxt("Phase must be single or double.");
    
    const mwSize *siz2 = mxGetDimensions(prhs[0]);
    const uint8_t *M2 = (const uint8_t*)mxGetData(prhs[1]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[2]);
    const mwSize *siz = mxGetDimensions(prhs[2]);
    
    if(is_s) {
        const float *x2 = (const float*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        float *x = (float*)mxGetData(plhs[0]);
        
        prolongf(x, Mask, x2, M2, siz, siz2);
    } else {
        const double *x2 = (const double*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        double *x = (double*)mxGetData(plhs[0]);
        
        prolongd(x, Mask, x2, M2, siz, siz2);
    }
    return;
}


void
prolongf(float *x, const uint8_t *Mask,
        const float *x2, const uint8_t *M2,
        const mwSize *siz, const mwSize *siz2)
{
    uint32_t i, j, k, l, l2;
    
    const uint32_t nx2   = (uint32_t)siz2[0];
    const uint32_t ny2   = (uint32_t)siz2[1];
    const uint32_t nz2   = (uint32_t)siz2[2];
    const uint32_t nxny2 = nx2*ny2;
    const uint32_t N2    = nxny2*nz2;
    const uint32_t NZ2   = nxny2*(nz2-1);
    const uint32_t NY2   = nx2*(ny2-1);
    const uint32_t NX2   = nx2-1;
    
    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nxny = nx*ny;
    const uint32_t nx22 = 2*nx;
    const uint32_t nxny22 = 2*nxny;
    
    uint32_t ll2[11], ll[4];
    ll[3]  = 1+nx+nxny;
    ll[2]  = nx+nxny;
    ll[1]  = 1+nxny;
    ll[0]  = 1+nx;
    ll2[0] = 1+nx2;
    ll2[1] = 1+nxny2;
    ll2[2] = nx2+nxny2;
    ll2[3] = 1+nx2+nxny2;
    
//     #pragma omp parallel for private(l, l2) collapse(3)
    for(k = 0; k < N2; k += nxny2) {
        for(j = 0; j < nxny2; j += nx2) {
            for(i = 0; i < nx2; ++i) {
                l  = (i << 1) + nx22*(j/nx2) + nxny22*(k/nxny2);
                
                if(Mask[l]) {
                    l2 = k + j + i;
                    x[l] = x2[l2];
                    
                    if(i != NX2 && Mask[l+1])
                        x[l+1] = M2[l2+1]
                        	? 0.5*(x2[l2] + x2[l2+1])
                            : x2[l2];
                            
                    if(j != NY2 && Mask[l+nx])
                        x[l+nx] = M2[l2+nx2]
                            ? 0.5*(x2[l2] + x2[l2+nx2])
                            : x2[l2];
                            
                    if(k != NZ2 && Mask[l+nxny])
                        x[l+nxny] = M2[l2+nxny2]
                            ? 0.5*(x2[l2] + x2[l2+nxny2])
                            : x2[l2];
                            
                    if(i != NX2 && j != NY2 & Mask[l+ll[0]])
                        x[l+ll[0]] = M2[l2+1] && M2[l2+nx2] && M2[l2+ll2[0]]
                        	? 0.25*(x2[l2] +
                            	x2[l2+1] + x2[l2+nx2] + x2[l2+ll2[0]])
                            : x2[l2];
                            
                    if(i != NX2 && k != NZ2 && Mask[l+ll[1]])       
                        x[l+ll[1]] = M2[l2+1] && M2[l2+nxny2] && M2[l2+ll2[1]]
                            ? 0.25*(x2[l2] +
                            	x2[l2+1] + x2[l2+nxny2] + x2[l2+ll2[1]])
                        	: x2[l2];
                            
                    if(j != NY2 && k != NZ2 && Mask[l+ll[2]])        
                        x[l+ll[2]] = M2[l2+nx2] && M2[l2+nxny2] && M2[l2+ll2[2]]
                         	? 0.25*(x2[l2] +
                            	x2[l2+nx2] + x2[l2+nxny2] + x2[l2+ll2[2]])
                            : x2[l2];
                            
                    if(i != NX2 && j != NY2 && k != NZ2 && Mask[l+ll[3]])        
                        x[l+ll[3]] = M2[l2+1] && M2[l2+nx2] && M2[l2+nxny2] 
                                && M2[l2+ll2[0]] && M2[l2+ll2[1]] 
                                && M2[l2+ll2[2]] && M2[l2+ll2[3]]
                        	? 0.125*(x2[l2] +
                            	x2[l2+1] + x2[l2+nx2] + x2[l2+nxny2] +
                                x2[l2+ll2[0]] + x2[l2+ll2[1]] + x2[l2+ll2[2]] +
                                x2[l2+ll2[3]])
                            : x2[l2];
                }
            }
        }
    }
    return;
}


void
prolongd(double *x, const uint8_t *Mask,
        const double *x2, const uint8_t *M2,
        const mwSize *siz, const mwSize *siz2)
{
    uint32_t i, j, k, l, l2;
     
    const uint32_t nx2   = (uint32_t)siz2[0];
    const uint32_t ny2   = (uint32_t)siz2[1];
    const uint32_t nz2   = (uint32_t)siz2[2];
    const uint32_t nxny2 = nx2*ny2;
    const uint32_t N2    = nxny2*nz2;
    const uint32_t NZ2   = nxny2*(nz2-1);
    const uint32_t NY2   = nx2*(ny2-1);
    const uint32_t NX2   = nx2-1;
    
    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nxny = nx*ny;
    const uint32_t nx22 = 2*nx;
    const uint32_t nxny22 = 2*nxny;
    
    uint32_t ll2[11], ll[4];
    ll[3]  = 1+nx+nxny;
    ll[2]  = nx+nxny;
    ll[1]  = 1+nxny;
    ll[0]  = 1+nx;
    ll2[0] = 1+nx2;
    ll2[1] = 1+nxny2;
    ll2[2] = nx2+nxny2;
    ll2[3] = 1+nx2+nxny2;
    
//     #pragma omp parallel for private(l, l2) collapse(3)
    for(k = 0; k < N2; k += nxny2) {
        for(j = 0; j < nxny2; j += nx2) {
            for(i = 0; i < nx2; ++i) {
                l  = (i << 1) + nx22*(j/nx2) + nxny22*(k/nxny2);
                
                if(Mask[l]) {
                    l2 = k + j + i;
                    x[l] = x2[l2];
                    
                    if(i != NX2 && Mask[l+1])
                        x[l+1] = M2[l2+1]
                        	? 0.5f*(x2[l2] + x2[l2+1])
                            : x2[l2];
                            
                    if(j != NY2 && Mask[l+nx])
                        x[l+nx] = M2[l2+nx2]
                            ? 0.5f*(x2[l2] + x2[l2+nx2])
                            : x2[l2];
                            
                    if(k != NZ2 && Mask[l+nxny])
                        x[l+nxny] = M2[l2+nxny2]
                            ? 0.5f*(x2[l2] + x2[l2+nxny2])
                            : x2[l2];
                            
                    if(i != NX2 && j != NY2 && Mask[l+ll[0]])
                        x[l+ll[0]] = M2[l2+1] && M2[l2+nx2] && M2[l2+ll2[0]]
                        	? 0.25f*(x2[l2] +
                            	x2[l2+1] + x2[l2+nx2] + x2[l2+ll2[0]])
                            : x2[l2];
                            
                    if(i != NX2 && k != NZ2 && Mask[l+ll[1]])       
                        x[l+ll[1]] = M2[l2+1] && M2[l2+nxny2] && M2[l2+ll2[1]]
                            ? 0.25f*(x2[l2] +
                            	x2[l2+1] + x2[l2+nxny2] + x2[l2+ll2[1]])
                        	: x2[l2];
                            
                    if(j != NY2 && k != NZ2 && Mask[l+ll[2]])        
                        x[l+ll[2]] = M2[l2+nx2] && M2[l2+nxny2] && M2[l2+ll2[2]]
                         	? 0.25f*(x2[l2] +
                            	x2[l2+nx2] + x2[l2+nxny2] + x2[l2+ll2[2]])
                            : x2[l2];
                            
                    if(i != NX2 && j != NY2 && k != NZ2 && Mask[l+ll[3]])        
                        x[l+ll[3]] = M2[l2+1] && M2[l2+nx2] && M2[l2+nxny2] 
                                && M2[l2+ll2[0]] && M2[l2+ll2[1]] 
                                && M2[l2+ll2[2]] && M2[l2+ll2[3]]
                        	? 0.125f*(x2[l2] +
                            	x2[l2+1] + x2[l2+nx2] + x2[l2+nxny2] +
                                x2[l2+ll2[0]] + x2[l2+ll2[1]] + x2[l2+ll2[2]] +
                                x2[l2+ll2[3]])
                            : x2[l2];
                }
            }
        }
    }
    return;
}
