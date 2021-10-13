#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* CONV_PER_D
 *
 * INPUT ARGUMENTS
 *  x: 3D complex double array
 *  g: 1D real convolution kernel
 *
 */

void conv_per_dim1(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi );
void conv_per_dim2(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi );
void conv_per_dim3(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi );

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* complex input array */
    uint32_t dim = (nrhs < 3) ? 1 : (uint32_t)((double*)mxGetData(prhs[2]))[0];
    
    /* complex input array */
    double *xr = (double*)mxGetData(prhs[0]);
    double *xi = (double*)mxGetImagData(prhs[0]);
    double *g  = (double*)mxGetData(prhs[1]);
    
    /* array and kernel sizes */
    const mwSize *xSize = mxGetDimensions(prhs[0]); //array size
    const mwSize *gSize = mxGetDimensions(prhs[1]); //kernel size
    const mwSize ndims  = mxGetNumberOfDimensions(prhs[0]);
	    
    /* complex output array */
    plhs[0] = mxCreateNumericArray(ndims, xSize, mxDOUBLE_CLASS, mxCOMPLEX);
    double *yr = (double*)mxGetData(plhs[0]);
    double *yi = (double*)mxGetImagData(plhs[0]);
    
    /* convolve arrays */
    if(dim == 1) {
        conv_per_dim1(xr,xi,g,xSize,ndims,gSize,yr,yi);
    } else if(dim == 2) {
        conv_per_dim2(xr,xi,g,xSize,ndims,gSize,yr,yi);
    } else if(dim == 3) {
        conv_per_dim3(xr,xi,g,xSize,ndims,gSize,yr,yi);
    }
    
    return;
}

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
    
#pragma omp parallel for collapse(2) private(i,l,l2,m) schedule(static) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            
            /* Inside Points */
            l2 = k + j; //"+len2-len2 = 0";
            l = l2 + len2;
            for(i = len2; i < nx-len2; ++i, ++l, ++l2) {
                yr[l] = xr[l2] * g[0];
                yi[l] = xi[l2] * g[0];
                for(m = 1; m < len; ++m) {
                    yr[l] += xr[l2+m] * g[m];
                    yi[l] += xi[l2+m] * g[m];
                }
            }
            
            /* First Points */
            l = k + j;
            l2 = l + nx;
            for(i = 0; i < len2; ++i, ++l, ++l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= i; ++m) {
                    yr[l] += xr[l+m] * g[len2+m] + xr[l-m] * g[len2-m];
                    yi[l] += xi[l+m] * g[len2+m] + xi[l-m] * g[len2-m];
                }
                for(m = i+1; m <= len2; ++m) {
                    yr[l] += xr[l+m] * g[len2+m] + xr[l2-m] * g[len2-m];
                    yi[l] += xi[l+m] * g[len2+m] + xi[l2-m] * g[len2-m];
                }
            }
            
            /* Last Points */
            l = k + j + NX;
            l2 = l - nx;
            for(i = NX; i > NX-len2; --i, --l, --l2) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1; m <= NX-i; ++m) {
                    yr[l] += xr[l+m] * g[len2+m] + xr[l-m] * g[len2-m];
                    yi[l] += xi[l+m] * g[len2+m] + xi[l-m] * g[len2-m];
                }
                for(m = NX-i+1; m <= len2; ++m) {
                    yr[l] += xr[l2+m] * g[len2+m] + xr[l-m] * g[len2-m];
                    yi[l] += xi[l2+m] * g[len2+m] + xi[l-m] * g[len2-m];
                }
            }
            
        }
    }
    
    return;
}

void conv_per_dim2(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi )
{
    uint32_t i, j, j2, k, l, l2, m, m2;
    
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
    const uint32_t nxlen2 = nx*len2;
    
#pragma omp parallel for collapse(2) num_threads(128)
    for(k = 0; k < nxnynz; k += nxny) {
        for(i = 0; i < nx; ++i) {
            
            /* Inside Points */
            l2 = k + i; //"+nx*len2-nx*len2 = 0";
            l = l2 + nxlen2;
            for(j = nxlen2; j < nxny-nxlen2; j += nx, l += nx, l2 += nx) {
                yr[l] = xr[l2] * g[0];
                yi[l] = xi[l2] * g[0];
                for(m = 1, m2 = nx; m < len; ++m, m2 += nx) {
                    yr[l] += xr[l2+m2] * g[m];
                    yi[l] += xi[l2+m2] * g[m];
                }
            }
            
            /* First Points */
            l = k + i;
            l2 = l + nxny;
            for(j = 0, j2 = 0; j < nxlen2; j += nx, ++j2, l += nx, l2 += nx) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1, m2 = nx; m <= j2; ++m, m2 += nx) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
                for(m = j2+1; m <= len2; ++m, m2 += nx) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l2-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l2-m2] * g[len2-m];
                }
            }
            
            /* Last Points */
            l = k + i + NY;
            l2 = l - nxny;
            for(j = NY, j2 = 0; j > NY-nxlen2; j -= nx, ++j2, l -= nx, l2 -= nx) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1, m2 = nx; m <= j2; ++m, m2 += nx) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
                for(m = j2+1; m <= len2; ++m, m2 += nx) {
                    yr[l] += xr[l2+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l2+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
            }
            
        }
    }
    
    return;
}

void conv_per_dim3(double *xr, double *xi, double *g, const mwSize *xSize, const mwSize ndims, const mwSize *gSize, double *yr, double *yi )
{
    uint32_t i, j, k, k2, l, l2, m, m2;
    
    const uint32_t nx       = (uint32_t)xSize[0];
    const uint32_t ny       = (uint32_t)xSize[1];
    const uint32_t nz       = (uint32_t)ndims == 3 ? (uint32_t)xSize[2] : 1;
    const uint32_t nxny     = nx*ny;
    const uint32_t nxnynz   = nxny*nz;
    const uint32_t NZ       = nxny*(nz-1);
    const uint32_t NY       = nx*(ny-1);
    const uint32_t NX       = nx-1;
    
    const uint32_t len      = (uint32_t)gSize[0];
    const uint32_t len2     = len/2;
    const uint32_t nxnylen2 = nxny*len2;
    
#pragma omp parallel for collapse(2) num_threads(128) //private(k,k2,l,l2,m,m2)
    for(j = 0; j < nxny; j += nx) {
        for(i = 0; i < nx; ++i) {
            
            /* Inside Points */
            l2 = j + i; //"+nx*len2-nx*len2 = 0";
            l = l2 + nxnylen2;
            for(k = nxnylen2; k < nxnynz-nxnylen2; k += nxny, l += nxny, l2 += nxny) {
                yr[l] = xr[l2] * g[0];
                yi[l] = xi[l2] * g[0];
                for(m = 1, m2 = nxny; m < len; ++m, m2 += nxny) {
                    yr[l] += xr[l2+m2] * g[m];
                    yi[l] += xi[l2+m2] * g[m];
                }
            }
            
            /* First Points */
            l = j + i;
            l2 = l + nxnynz;
            for(k = 0, k2 = 0; k < nxnylen2; k += nxny, ++k2, l += nxny, l2 += nxny) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1, m2 = nxny; m <= k2; ++m, m2 += nxny) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
                for(m = k2+1; m <= len2; ++m, m2 += nxny) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l2-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l2-m2] * g[len2-m];
                }
            }
            
            /* Last Points */
            l = j + i + NZ;
            l2 = l - nxnynz;
            for(k = NZ, k2 = 0; k > NZ-nxnylen2; k -= nxny, ++k2, l -= nxny, l2 -= nxny) {
                yr[l] = xr[l] * g[len2];
                yi[l] = xi[l] * g[len2];
                for(m = 1, m2 = nxny; m <= k2; ++m, m2 += nxny) {
                    yr[l] += xr[l+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
                for(m = k2+1; m <= len2; ++m, m2 += nxny) {
                    yr[l] += xr[l2+m2] * g[len2+m] + xr[l-m2] * g[len2-m];
                    yi[l] += xi[l2+m2] * g[len2+m] + xi[l-m2] * g[len2-m];
                }
            }
            
        }
    }
    
    return;
}

