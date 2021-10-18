#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>

/* FMG_DIFFUSE_CD
 *
 * INPUT ARGUMENTS
 *  x: input array (3D complex double array)
 *  h: grid spacing (scalar double)
 *  D: diffusion constant (scalar double)
 *  f: decay term (3D complex double array)
 *  c: arbitrary real constant (scalar double)
 */

/* Simple aliases for input pointers */
#define __x__  prhs[0]
#define __h__  prhs[1]
#define __D__  prhs[2]
#define __f__  prhs[3]
#define __c__  prhs[4]

/* Simple aliases for output pointers */
#define __dx__ plhs[0]

#define USE_PARALLEL 1

void fmg_diffuse3D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double c, const mwSize *mSize );
void fmg_diffuse4D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double c, const mwSize *mSize );

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* grid spacing h */
    double h = (nrhs < 2) ? 1.0 : ((double*)mxGetData(__h__))[0];
    
    /* diffusion constant D */
    double D = (nrhs < 3) ? 1.0 : ((double*)mxGetData(__D__))[0];
    
    /* Arbitrary real constant c */
    const double c = (nrhs < 5) ? 1.0 : ((double*)mxGetData(__c__))[0];
    
    /* Normalized and scaled diffusion constant K = c*D/h^2 */
    const double K = (c*D)/(h*h);
    
    /* input array size */
    const mwSize  ndim  = mxGetNumberOfDimensions(__x__);
    const mwSize *mSize = mxGetDimensions(__x__);
    
    /* complex input array */
    const double *xr = (const double *)mxGetData(__x__);
    const double *xi = (const double *)mxGetImagData(__x__);
    
    /* complex decay term */
    const double *fr = (const double*)mxGetData(__f__);
    const double *fi = (const double*)mxGetImagData(__f__);
    
    /* complex output array */
    __dx__ = mxCreateNumericArray(ndim, mSize, mxDOUBLE_CLASS, mxCOMPLEX);
    double *dxr = (double*)mxGetData(__dx__);
    double *dxi = (double*)mxGetImagData(__dx__);
        
    if( ndim == 3 ) {
        fmg_diffuse3D( dxr, dxi, xr, xi, fr, fi, K, c, mSize );
    } else {
        fmg_diffuse4D( dxr, dxi, xr, xi, fr, fi, K, c, mSize );
    }
    
    return;

}

void fmg_diffuse3D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double c, const mwSize *mSize ) {
    
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;
    
// #if USE_PARALLEL
// #pragma omp parallel for collapse(2) schedule(static) private(i,l,il,ir,jl,jr,kl,kr) num_threads(128)
// #endif
//     for(k = 0; k < nxnynz; k += nxny) {
//         for(j = 0; j < nxny; j += nx) {
//             l = k + j;
//             for(i = 0; i < nx; ++i, ++l) {
//                 il = (i==0 ) ? l+NX : l-1;
//                 ir = (i==NX) ? l-NX : l+1;
//                 jl = (j==0 ) ? l+NY : l-nx;
//                 jr = (j==NY) ? l-NY : l+nx;
//                 kl = (k==0 ) ? l+NZ : l-nxny;
//                 kr = (k==NZ) ? l-NZ : l+nxny;
//                 
//                 dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
//                 dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
//             }
//         }
//     }
    
#if USE_PARALLEL
#pragma omp parallel for collapse(2) //schedule(static)//num_threads(128)
#endif
    for(k = nxny; k < nxnynz-nxny; k += nxny) {
        for(j = nx; j < nxny-nx; j += nx) {
            l = k + j + 1;
            for(i = 1; i < nx-1; ++i, ++l) {
                dxr[l] = K * (-6.0 * xr[l] + xr[l-1] + xr[l+1] + xr[l-nx] + xr[l+nx] + xr[l-nxny] + xr[l+nxny]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
                dxi[l] = K * (-6.0 * xi[l] + xi[l-1] + xi[l+1] + xi[l-nx] + xi[l+nx] + xi[l-nxny] + xi[l+nxny]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
            }
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(i,j,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif
    for(j = 0; j < nxny; j += nx) {
        for(i = 0; i < nx; ++i) {
            l  = j + i;
            il = (i==0 ) ? l+NX : l-1;
            ir = (i==NX) ? l-NX : l+1;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            
            kl = l+NZ;
            kr = l+nxny;
            
            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
            
            l  += NZ;
            il += NZ;
            ir += NZ;
            jl += NZ;
            jr += NZ;
            
            kl = l-nxny;
            kr = l-NZ;
            
            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(i,k,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif
    for(k = 0; k < nxnynz; k += nxny) {
        for(i = 0; i < nx; ++i) {
            l = k + i;
            il = (i==0 ) ? l+NX : l-1;
            ir = (i==NX) ? l-NX : l+1;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;
            
            jl = l+NY;
            jr = l+nx;
            
            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
            
            l  += NY;
            il += NY;
            ir += NY;
            kl += NY;
            kr += NY;
            
            jl = l-nx;
            jr = l-NY;
            
            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(j,k,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l  = k + j;
            
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;
            
            il = l+NX;
            ir = l+1;
            
            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );

            l  += NX;
            jl += NX;
            jr += NX;
            kl += NX;
            kr += NX;

            il = l-1;
            ir = l-NX;

            dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
            dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
        }
    }
    
    return;
}

void fmg_diffuse4D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double c, const mwSize *mSize ) {
    
    const uint32_t nx       = (uint32_t)mSize[0];
    const uint32_t ny       = (uint32_t)mSize[1];
    const uint32_t nz       = (uint32_t)mSize[2];
    const uint32_t nw       = (uint32_t)mSize[3];
    const uint32_t nxny     = nx*ny;
    const uint32_t nxnynz   = nxny*nz;
    const uint32_t nxnynznw = nxnynz*nw;
    const uint32_t NX       = nx-1;
    const uint32_t NY       = nx*(ny-1);
    const uint32_t NZ       = nxny*(nz-1);
    const uint32_t NW       = nxnynz*(nw-1);
    
    uint32_t i, j, k, w, l, il, ir, jl, jr, kl, kr;
    
#if USE_PARALLEL
#pragma omp parallel for collapse(3) num_threads(128)
#endif
    for(w = 0; w < nxnynznw; w += nxnynz) {
        for(k = 0; k < nxnynz; k += nxny) {
            for(j = 0; j < nxny; j += nx) {
                l = k + j + w;
                for(i = 0; i < nx; ++i, ++l) {
                    il = (i==0 ) ? l+NX : l-1;
                    ir = (i==NX) ? l-NX : l+1;
                    jl = (j==0 ) ? l+NY : l-nx;
                    jr = (j==NY) ? l-NY : l+nx;
                    kl = (k==0 ) ? l+NZ : l-nxny;
                    kr = (k==NZ) ? l-NZ : l+nxny;
                    
                    dxr[l] = K * (-6.0 * xr[l] + xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - c * ( fr[l] * xr[l] - fi[l] * xi[l] );
                    dxi[l] = K * (-6.0 * xi[l] + xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - c * ( fr[l] * xi[l] + fi[l] * xr[l] );
                }
            }
        }
    }
    
    return;
}