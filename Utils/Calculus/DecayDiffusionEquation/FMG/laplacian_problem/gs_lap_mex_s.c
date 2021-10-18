#include <stdint.h>
#include <math.h>
#include "matrix.h"
#include "mex.h"


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//     plhs[0]  = mxDuplicateArray((mxArray*)prhs[0]);
    plhs[0]  = (mxArray*)prhs[0];
    mxUnshareArray(plhs[0], true);
    float *v = (float*)mxGetData(plhs[0]);
    
    const float *b   = (const float*)mxGetData(prhs[1]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[2]);
    uint32_t maxIter = (uint32_t)mxGetScalar(prhs[3]);
        
    float h2 = 1.0f;
    if(nrhs > 4) {
        double *vSize = (double*)mxGetData(prhs[4]);
        h2 = (float)(vSize[0]*vSize[0]);
    }
   
    const mwSize *mSize = mxGetDimensions(prhs[0]);
    const uint32_t nx   = (uint32_t)mSize[0];
    const uint32_t ny   = (uint32_t)mSize[1];
    const uint32_t nz   = (uint32_t)mSize[2];
    const uint32_t nxny = nx*ny;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
   
    uint8_t IN_Z, IN_ZY, IN_M;
    uint32_t i, j, k, l;

    while(maxIter--) {
        for(k = 0; k < nxny*nz; k += nxny) {
            IN_Z = (k != 0) & (k != NZ);

            for(j = 0; j < nxny; j += nx) {
                IN_ZY = IN_Z & (j != 0) & (j != NY);
                l = k + j;
                
                for(i = 0; i < nx; ++i, ++l) {
                    if(IN_ZY & (i != 0) & (i != NX)) {
                        IN_M = Mask [l]
                            & Mask[l-1] & Mask[l+1]
                            & Mask[l-nx] & Mask[l+nx]
                            & Mask[l-nxny] & Mask[l+nxny];
                        v[l] = IN_M
                            ? 0.1666667f *
                              (v[l-1] + v[l+1] +
                               v[l-nx] + v[l+nx] + 
                               v[l-nxny] + v[l+nxny] - h2*b[l])
                            : 0.0f;
                    } else {
                        v[l] = 0.0f;
                    }
                }
            }
        }
    }

    return;
    
}
