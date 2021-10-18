#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* CONV_EVEN_PER_D
 *
 * INPUT ARGUMENTS
 *  x: 3D real double array
 *  g: 1D real kernel, assumed to be even about center with odd length
 *
 */

void conv_per_dim1(double *x, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *y );

void
        mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* complex input array */
    uint32_t dim = (nrhs < 3) ? 1 : (uint32_t)((double*)mxGetData(prhs[2]))[0];
    
    /* complex input array */
    double *x = (double*)mxGetData(prhs[0]);
    double *g  = (double*)mxGetData(prhs[1]);
    
    /* array and kernel sizes */
    const mwSize *xSize = mxGetDimensions(prhs[0]); //array size
    const mwSize *gSize = mxGetDimensions(prhs[1]); //kernel size
    const mwSize ndims  = mxGetNumberOfDimensions(prhs[0]);
    
    /* complex output array */
    plhs[0] = mxCreateNumericArray(ndims, xSize, mxDOUBLE_CLASS, mxREAL);
    double *y = (double*)mxGetData(plhs[0]);
    
    /* convolve arrays */
    conv_per_dim1(x,g,xSize,ndims,gSize,y);
    
    return;
}

void conv_per_dim1(double *x, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *y )
{
    uint32_t i, j, k, l, l2, m;
    
    const uint32_t nx     = (uint32_t)xSize[0];
    const uint32_t ny     = (uint32_t)xSize[1];
    const uint32_t nz     = (uint32_t)ndims == 3 ? (uint32_t)xSize[2] : 1;
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NZ     = nxny*(nz-1);
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NX     = nx-1;
    
    const uint32_t len    = (uint32_t)gSize[0];
    const uint32_t len2   = len/2;
    
    double g0[len];
    for(m = 0; m<len; ++m) {
        g0[m] = m == len2 ? 0.5 * g[m] : g[m];
    }
    
    /* Inside Points */
#pragma omp parallel for collapse(3) private(i,l) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(m = 0; m <= len2; ++m) {
                double G = g0[m];
                l = k + j + len2;
                for(i = len2; i < nx-len2; ++i, ++l) {
                    y[l] += (x[l+len2-m] + x[l-len2+m]) * G;
                }
            }
        }
    }
    
    /* First and Last Points */
#pragma omp parallel for collapse(2) private(i,l,l2,m) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            
            /* First Points */
            l = k + j;
            l2 = l + nx;
            for(i = 0; i < len2; ++i, ++l, ++l2) {
                y[l] = x[l] * g[len2];
                for(m = 1; m <= i; ++m) {
                    y[l] += (x[l+m] + x[l-m]) * g[len2+m];
                }
                for(m = i+1; m <= len2; ++m) {
                    y[l] += (x[l+m] + x[l2-m]) * g[len2+m];
                }
            }
            
            /* Last Points */
            l = k + j + NX;
            l2 = l - nx;
            for(i = NX; i > NX-len2; --i, --l, --l2) {
                y[l] = x[l] * g[len2];
                for(m = 1; m <= NX-i; ++m) {
                    y[l] += (x[l+m] + x[l-m]) * g[len2+m];
                }
                for(m = NX-i+1; m <= len2; ++m) {
                    y[l] += (x[l2+m] + x[l-m]) * g[len2+m];
                }
            }
            
        }
    }
    
    return;
}

