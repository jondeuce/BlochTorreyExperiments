#include <stdint.h>
#include <math.h>
#include "mex.h"


void gsf(float *v, const float *b, const uint8_t *Mask,
        double *h, uint32_t maxIter, const mwSize *siz);

void gsd(double *v, const double *b, const uint8_t *Mask,
        double *h, uint32_t maxIter, const mwSize *siz);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 3)
        mexErrMsgTxt("Initial guess v, b and Mask needed.");
    
    if(!mxIsLogical(prhs[2]))
        mexErrMsgTxt("Mask must be logical.");
    
    const uint8_t is_s = mxIsSingle(prhs[0]);
    const uint8_t is_d = mxIsDouble(prhs[0]);
    
    if(!is_s && !is_d)
        mexErrMsgTxt("v must be single or double.");
    
    if( (is_s && !mxIsSingle(prhs[1])) || (is_d && !mxIsDouble(prhs[1])) )
        mexErrMsgTxt("v and b must have same precision.");
    
    const mwSize *siz = mxGetDimensions(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[2]);
    uint32_t maxIter = nrhs > 3 ? (uint32_t)mxGetScalar(prhs[3]) : 1;
    
    double *h = (double[3]){1.0, 1.0, 1.0};
    if(nrhs > 4) {
        const double *hh = (const double*)mxGetData(prhs[4]);
        h = mxGetNumberOfElements(prhs[4]) < 3
        	? (double[3]){hh[0]*hh[0], hh[0]*hh[0], hh[0]*hh[0]}
            : (double[3]){hh[0]*hh[0], hh[1]*hh[1], hh[2]*hh[2]};
    }
    
// 	plhs[0]  = mxDuplicateArray((mxArray*)prhs[0]);
    plhs[0] = (mxArray*)prhs[0];
    mxUnshareArray(plhs[0], true);
        
    if(is_s) {
        float *v = (float*)mxGetData(plhs[0]);
        const float *b = (const float*)mxGetData(prhs[1]);
        
        gsf(v, b, Mask, h, maxIter, siz);
    } else {
        double *v = (double*)mxGetData(plhs[0]);
        const double *b = (const double*)mxGetData(prhs[1]);
        
        gsd(v, b, Mask, h, maxIter, siz);
    }
    return;
}


void
gsf(float *v, const float *b, const uint8_t *Mask,
        double *h, uint32_t maxIter, const mwSize *siz)
{
    uint32_t i, j, k, l;
    
    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    const uint32_t NZZ  = NZ-nxny;
    const uint32_t NYY  = NY-nx;
    const uint32_t NXX  = NX-1;
    
    float hx = (float)(h[1]*h[2]);
    float hy = (float)(h[0]*h[2]);
    float hz = (float)(h[0]*h[1]);
    float hb = (float)(h[0]*h[1]*h[2]);
    float hh = 1.0f / (2.0f*(hx+hy+hz));
    
    hx = hh*hx;
    hy = hh*hy;
    hz = hh*hz;
    hb = hh*hb;
    
    for(k = 0; k < nxny*nz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = k + j;
            for(i = 0; i < nx; ++i, ++l) {
                if( Mask[l] &&
                        (i-1 < NXX) && Mask[l-1] && Mask[l+1] &&
                        (j-nx < NYY) && Mask[l-nx] && Mask[l+nx] &&
                        (k-nxny < NZZ) && Mask[l-nxny] && Mask[l+nxny] ) {

                    v[l] = hy*(v[l-1] + v[l+1]) +
                           hx*(v[l-nx] + v[l+nx]) +
                           hz*(v[l-nxny] + v[l+nxny]) - hb*b[l];
                } else {
                    v[l] = 0.0f;
                }
            }
        }
    }
    
    while(--maxIter) {
        for(k = nxny; k < NZ; k += nxny) {
            for(j = nx; j < NY; j += nx) {
                l = k + j + 1;
                for(i = 1; i < NX; ++i, ++l) {
                    if( Mask[l] &&
                            Mask[l-1] && Mask[l+1] &&
                            Mask[l-nx] && Mask[l+nx] &&
                            Mask[l-nxny] && Mask[l+nxny] ) {
                        
                        v[l] = hy*(v[l-1] + v[l+1]) +
                               hx*(v[l-nx] + v[l+nx]) +
                               hz*(v[l-nxny] + v[l+nxny]) - hb*b[l];
                    }
                }
            }
        }
    }
    return;
}


void
gsd(double *v, const double *b, const uint8_t *Mask,
        double *h, uint32_t maxIter, const mwSize *siz)
{
    uint32_t i, j, k, l;
    
    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    const uint32_t NZZ  = NZ-nxny;
    const uint32_t NYY  = NY-nx;
    const uint32_t NXX  = NX-1;
    
    double hx = h[1]*h[2];
    double hy = h[0]*h[2];
    double hz = h[0]*h[1];
    double hb = h[0]*h[1]*h[2];
    double hh = 1.0 / (2.0*(hx+hy+hz));
    
    hx = hh*hx;
    hy = hh*hy;
    hz = hh*hz;
    hb = hh*hb;
    
    for(k = 0; k < nxny*nz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = k + j;
            for(i = 0; i < nx; ++i, ++l) {
                if( Mask[l] &&
                        (i-1 < NXX) && Mask[l-1] && Mask[l+1] &&
                        (j-nx < NYY) && Mask[l-nx] && Mask[l+nx] &&
                        (k-nxny < NZZ) && Mask[l-nxny] && Mask[l+nxny] ) {

                    v[l] = hy*(v[l-1] + v[l+1]) +
                           hx*(v[l-nx] + v[l+nx]) +
                           hz*(v[l-nxny] + v[l+nxny]) - hb*b[l];
                } else {
                    v[l] = 0.0;
                }
            }
        }
    }
    
    while(--maxIter) {
        for(k = nxny; k < NZ; k += nxny) {
            for(j = nx; j < NY; j += nx) {
                l = k + j + 1;
                for(i = 1; i < NX; ++i, ++l) {
                    if( Mask[l] &&
                            Mask[l-1] && Mask[l+1] &&
                            Mask[l-nx] && Mask[l+nx] &&
                            Mask[l-nxny] && Mask[l+nxny] ) {
                        
                        v[l] = hy*(v[l-1] + v[l+1]) +
                               hx*(v[l-nx] + v[l+nx]) +
                               hz*(v[l-nxny] + v[l+nxny]) - hb*b[l];
                    }
                }
            }
        }
    }
    return;
}
