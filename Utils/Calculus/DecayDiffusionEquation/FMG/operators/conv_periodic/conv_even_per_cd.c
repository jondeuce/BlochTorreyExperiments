#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* CONV_EVEN_PER_CD
 *
 * INPUT ARGUMENTS
 *  x: 3D complex double array
 *  g: 1D real kernel, assumed to be even about center with odd length
 *
 */

void conv_per_dim1(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi );


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // complex input array
    uint32_t dim = (nrhs < 3) ? 1 : (uint32_t)((double*)mxGetData(prhs[2]))[0];
    
    // complex input array
    double *xr = (double*)mxGetData(prhs[0]);
    double *xi = (double*)mxGetImagData(prhs[0]);
    double *g  = (double*)mxGetData(prhs[1]);
    
    // array and kernel sizes
    const mwSize *xSize = mxGetDimensions(prhs[0]); //array size
    const mwSize *gSize = mxGetDimensions(prhs[1]); //kernel size
    const mwSize ndims  = mxGetNumberOfDimensions(prhs[0]);
    
    // complex output array
    plhs[0] = mxCreateNumericArray(ndims, xSize, mxDOUBLE_CLASS, mxCOMPLEX);
    double *yr = (double*)mxGetData(plhs[0]);
    double *yi = (double*)mxGetImagData(plhs[0]);
    
    // convolve arrays
    conv_per_dim1(xr,xi,g,xSize,ndims,gSize,yr,yi);
    
    return;
}

/**************************************************************************
 * This version is faster on curie
 *************************************************************************/
void conv_per_dim1(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi )
{
    uint32_t i, j, k, l, l2, l3, m;
    
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
    
    double g0[len], G;
    for(m = 0; m<len; ++m) {
        g0[m] = m == len2 ? 0.5 * g[m] : g[m];
    }
    
#pragma omp parallel for collapse(2) private(i,m,l,l2,l3) schedule(static) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            
            //Inside Points
            l = k + j + len2;
            l2 = l - len2;
            l3 = l + len2;
            for(i = len2; i < nx-len2; ++i, ++l, ++l2, ++l3) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 0; m < len2; ++m) {
                    yr[l] += (xr[l3-m] + xr[l2+m]) * g0[m];
                    yi[l] += (xi[l3-m] + xi[l2+m]) * g0[m];
                }
            }
//             for(m = 0; m <= len2; ++m) {
//                 l = k + j + len2;
//                 l2 = l - len2 + m;
//                 l3 = l + len2 - m;
//                 for(i = len2; i < nx-len2; ++i, ++l, ++l2, ++l3) {
//                     yr[l] += (xr[l3] + xr[l2]) * g0[m];
//                     yi[l] += (xi[l3] + xi[l2]) * g0[m];
//                 }
//             }
                        
            /* First Points */
            l = k + j;
            l2 = l + nx;
            for(i = 0; i < len2; ++i, ++l, ++l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= i; ++m) {
                    yr[l] += (xr[l+m] + xr[l-m]) * g0[len2+m];
                    yi[l] += (xi[l+m] + xi[l-m]) * g0[len2+m];
                }
                for(m = i+1; m <= len2; ++m) {
                    yr[l] += (xr[l+m] + xr[l2-m]) * g0[len2+m];
                    yi[l] += (xi[l+m] + xi[l2-m]) * g0[len2+m];
                }
            }
            
            /* Last Points */
            l = k + j + NX;
            l2 = l - nx;
            for(i = NX; i > NX-len2; --i, --l, --l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= NX-i; ++m) {
                    yr[l] += (xr[l+m] + xr[l-m]) * g0[len2-m];
                    yi[l] += (xi[l+m] + xi[l-m]) * g0[len2-m];
                }
                for(m = NX-i+1; m <= len2; ++m) {
                    yr[l] += (xr[l2+m] + xr[l-m]) * g0[len2-m];
                    yi[l] += (xi[l2+m] + xi[l-m]) * g0[len2-m];
                }
            }
            
        }
    }
    
    return;
}

/**************************************************************************
 * This version is faster on penelope
 *************************************************************************/
/*
void conv_per_dim1(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi )
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
    
    //Inside Points
#pragma omp parallel for collapse(3) private(i,l) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(m = 0; m <= len2; ++m) {
                double G = g0[m];
                l = k + j + len2;
                for(i = len2; i < nx-len2; ++i, ++l) {
                    yr[l] += (xr[l+len2-m] + xr[l-len2+m]) * G;
                    yi[l] += (xi[l+len2-m] + xi[l-len2+m]) * G;
                }
            }
        }
    }
    
    //First and Last Points
#pragma omp parallel for collapse(2) private(i,l,l2,m) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            
            //First Points
            l = k + j;
            l2 = l + nx;
            for(i = 0; i < len2; ++i, ++l, ++l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= i; ++m) {
                    yr[l] += (xr[l+m] + xr[l-m]) * g[len2+m];
                    yi[l] += (xi[l+m] + xi[l-m]) * g[len2+m];
                }
                for(m = i+1; m <= len2; ++m) {
                    yr[l] += (xr[l+m] + xr[l2-m]) * g[len2+m];
                    yi[l] += (xi[l+m] + xi[l2-m]) * g[len2+m];
                }
            }
            
            //Last Points
            l = k + j + NX;
            l2 = l - nx;
            for(i = NX; i > NX-len2; --i, --l, --l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= NX-i; ++m) {
                    yr[l] += (xr[l+m] + xr[l-m]) * g[len2+m];
                    yi[l] += (xi[l+m] + xi[l-m]) * g[len2+m];
                }
                for(m = NX-i+1; m <= len2; ++m) {
                    yr[l] += (xr[l2+m] + xr[l-m]) * g[len2+m];
                    yi[l] += (xi[l2+m] + xi[l-m]) * g[len2+m];
                }
            }
            
        }
    }
    
    return;
}
*/