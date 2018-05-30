#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"

/* TRANS_BLOCHTORREYACTION_CD
 *
 * INPUT ARGUMENTS
 *  prhs[0] -> x:     input array (3D complex double array)
 *  prhs[1] -> h:     grid spacing (scalar double)
 *  prhs[2] -> D:     diffusion constant (scalar double)
 *  prhs[3] -> f:     diagonal term: combined Gamma and -6*D/h^2 (3D complex double array)
 *  prhs[4] -> gsize: size of grid operated on (3 or 4 element double array)
 *  prhs[5] -> ndim:  number of dimensions operated on (scalar double = 3 or 4)
 *  prhs[6] -> iters: number of iterations to apply the trans_BlochTorreyAction (scalar double)
 *
 * OUTPUT ARGUMENTS
 *  plhs[0] -> dx:    output array (3D complex double array)
 *
 */

/* Simple aliases for input pointers */
#define __x__      (prhs[0])
#define __h__      (prhs[1])
#define __D__      (prhs[2])
#define __f__      (prhs[3])
#define __gsize__  (prhs[4])
#define __ndim__   (prhs[5])
#define __iters__  (prhs[6])

/* Simple aliases for output pointers */
#define __dx__     (plhs[0])

/* Flag for using omp parallel for loops */
#define USE_PARALLEL 1

/* Enum for different implementations (Type 2 currently fastest) */
#define TRANS_BLOCHTORREY3D_TYPE 2

/* Simple unsafe swap macro: https://stackoverflow.com/questions/3982348/implement-generic-swap-macro-in-c */
#define SWAP(x,y) do { typeof(x) SWAP = x; x = y; y = SWAP; } while (0)

void trans_BlochTorreyAction3D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double *gsize );
void trans_BlochTorreyAction4D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double *gsize );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* grid spacing h */
    const double h = ((const double*)mxGetData(__h__))[0];
    
    /* diffusion constant D */
    const double D = ((const double*)mxGetData(__D__))[0];
    
    /* Normalized and scaled diffusion constant K = D/h^2 */
    const double K = D/(h*h);
    
    /* Represented dimensions of input: ndim will be 3 or 4, and gsize will have respectively 3 or 4 elements */
    const double  *gsize = (const double*)mxGetData(__gsize__);
    const uint32_t ndim  = (const uint32_t) ((double*)mxGetData(__ndim__))[0];
    
    /* Number of iterations to apply */
    const uint32_t iters = (const uint32_t) ((double*)mxGetData(__iters__))[0];
    
    /* Actual dimensions of input: want to support 'flattened' 3D -> 1D, as well as full 3D */
    const mwSize *xsize = mxGetDimensions(__x__);
    const mwSize  xdim  = mxGetNumberOfDimensions(__x__);
    
    /* complex input array */
    const double *xr = (const double*)mxGetData(__x__);
    const double *xi = (const double*)mxGetImagData(__x__);
    
    /* Dummy temp variable (needed for multiple iterations) */
    double *yr, *yi;
    
    if( iters > 1 ) {
        /* If we are doing multiple applications, we must make a copy to 
         * store the temp. Consider e.g. y = A*(A*x) -> z = A*x0; y = A*z; */
        yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(double));
        yi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(double));
    }
    
    /* complex diagonal term */
    const double *fr = (const double*)mxGetData(__f__);
    const double *fi = (const double*)mxGetImagData(__f__);
    
    /* temporary variable which will be later associated with the complex output array */
    double *dxr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(double));
    double *dxi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(double));
    
    void (*trans_BlochTorreyAction)(double *, double *, const double *, const double *, const double *, const double *, const double, const double *);
    if( ndim == 3 )
        trans_BlochTorreyAction = &trans_BlochTorreyAction3D;
    else
        trans_BlochTorreyAction = &trans_BlochTorreyAction4D;
    
    /* Evaluate the trans_BlochTorreyAction once with input data */
    trans_BlochTorreyAction( dxr, dxi, xr, xi, fr, fi, K, gsize );
    
    /* Evaluate the trans_BlochTorreyAction iters-1 times using temp variable, if necessary*/
    int i;
    for(i = 1; i < iters; ++i) {
        SWAP(dxr, yr);
        SWAP(dxi, yi);
        trans_BlochTorreyAction( dxr, dxi, yr, yi, fr, fi, K, gsize );
    }
    
    /* Create complex output array */
    __dx__ = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxCOMPLEX); /* Create an empty array */
    mxSetDimensions(__dx__, mxGetDimensions(__x__), mxGetNumberOfDimensions(__x__)); /* Set the dimensions to be same as input */
    
    /* Associate with output array */
    mxSetData(__dx__, dxr); /* Assign real part */
    mxSetImagData(__dx__, dxi); /* Assign imag part */
    
    /* Free temporary variable, if necessary */
    if( iters > 1 ) {
        mxFree(yr);
        mxFree(yi);
    }
    
    return;
}

void trans_BlochTorreyAction3D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double *gsize ) {
    
    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;
    
    
    /* *******************************************************************
     * TRANS_BLOCHTORREY3D_TYPE = 1: Triply-nested for-loop, twice collapsed
     ******************************************************************* */
#if (TRANS_BLOCHTORREY3D_TYPE == 1)
    
#if USE_PARALLEL
#pragma omp parallel for collapse(2) num_threads(128) schedule(static) private(i,l,il,ir,jl,jr,kl,kr)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l = k + j;
            for(i = 0; i < nx; ++i, ++l) {
                il = (i==0 ) ? l+NX : l-1;
                ir = (i==NX) ? l-NX : l+1;
                jl = (j==0 ) ? l+NY : l-nx;
                jr = (j==NY) ? l-NY : l+nx;
                kl = (k==0 ) ? l+NZ : l-nxny;
                kr = (k==NZ) ? l-NZ : l+nxny;
                
                dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
                dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            }
        }
    }
    
#endif /* (TRANS_BLOCHTORREY3D_TYPE == 1) */
    
    
    /* *******************************************************************
     * TRANS_BLOCHTORREY3D_TYPE = 2:
     *     One triply-nested loops for inner points (no BC checking), then
     *     three doubly-nested loops for boundaries
     ******************************************************************* */
#if (TRANS_BLOCHTORREY3D_TYPE == 2)
    
#if USE_PARALLEL
#pragma omp parallel for collapse(2) //schedule(static)//num_threads(128)
#endif /* USE_PARALLEL */
    for(k = nxny; k < nxnynz-nxny; k += nxny) {
        for(j = nx; j < nxny-nx; j += nx) {
            l = k + j + 1;
            for(i = 1; i < nx-1; ++i, ++l) {
                dxr[l] = K * (xr[l-1] + xr[l+1] + xr[l-nx] + xr[l+nx] + xr[l-nxny] + xr[l+nxny]) + (fr[l] * xr[l] + fi[l] * xi[l]);
                dxi[l] = K * (xi[l-1] + xi[l+1] + xi[l-nx] + xi[l+nx] + xi[l-nxny] + xi[l+nxny]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            }
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(i,j,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif /* USE_PARALLEL */
    for(j = 0; j < nxny; j += nx) {
        for(i = 0; i < nx; ++i) {
            l  = j + i;
            il = (i==0 ) ? l+NX : l-1;
            ir = (i==NX) ? l-NX : l+1;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            
            kl = l+NZ;
            kr = l+nxny;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            
            l  += NZ;
            il += NZ;
            ir += NZ;
            jl += NZ;
            jr += NZ;
            
            kl = l-nxny;
            kr = l-NZ;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(i,k,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(i = 0; i < nx; ++i) {
            l = k + i;
            il = (i==0 ) ? l+NX : l-1;
            ir = (i==NX) ? l-NX : l+1;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;
            
            jl = l+NY;
            jr = l+nx;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            
            l  += NY;
            il += NY;
            ir += NY;
            kl += NY;
            kr += NY;
            
            jl = l-nx;
            jr = l-NY;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for //schedule(static) //collapse(2) //private(j,k,l,il,ir,jl,jr,kl,kr) //num_threads(128)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            l  = k + j;
            
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;
            
            il = l+NX;
            ir = l+1;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            
            l  += NX;
            jl += NX;
            jr += NX;
            kl += NX;
            kr += NX;
            
            il = l-1;
            ir = l-NX;
            
            dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
            dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
        }
    }
    
#endif /* (TRANS_BLOCHTORREY3D_TYPE == 2) */
    
    
    /* *******************************************************************
     * TRANS_BLOCHTORREY3D_TYPE = 3: Triply-nested for-loop, thrice collapsed
     ******************************************************************* */
#if (TRANS_BLOCHTORREY3D_TYPE == 3)
    
#if USE_PARALLEL
#pragma omp parallel for collapse(3) num_threads(128) schedule(static) private(i,l,il,ir,jl,jr,kl,kr)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                il = (i==0 ) ? l+NX : l-1;
                ir = (i==NX) ? l-NX : l+1;
                jl = (j==0 ) ? l+NY : l-nx;
                jr = (j==NY) ? l-NY : l+nx;
                kl = (k==0 ) ? l+NZ : l-nxny;
                kr = (k==NZ) ? l-NZ : l+nxny;
                
                dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
                dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            }
        }
    }
    
#endif /* (TRANS_BLOCHTORREY3D_TYPE == 3) */
    
    
    /* *******************************************************************
     * TRANS_BLOCHTORREY3D_TYPE = 4:
     *     Three triply-nested for-loops, splitting x/y/z indexing into
     *     separate loops for cache-locality
     ******************************************************************* */
#if (TRANS_BLOCHTORREY3D_TYPE == 4)
    
#if USE_PARALLEL
#pragma omp parallel for collapse(3) num_threads(128) schedule(static) private(i,l,il,ir,jl,jr,kl,kr)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                il = (i==0 ) ? l+NX : l-1;
                ir = (i==NX) ? l-NX : l+1;
                
                dxr[l] = K * (xr[il] + xr[ir]) + (fr[l] * xr[l] + fi[l] * xi[l]);
                dxi[l] = K * (xi[il] + xi[ir]) + (fr[l] * xi[l] - fi[l] * xr[l]);
            }
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for collapse(3) num_threads(128) schedule(static) private(i,l,il,ir,jl,jr,kl,kr)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                jl = (j==0 ) ? l+NY : l-nx;
                jr = (j==NY) ? l-NY : l+nx;
                
                dxr[l] += K * (xr[jl] + xr[jr]);
                dxi[l] += K * (xi[jl] + xi[jr]);
            }
        }
    }
    
#if USE_PARALLEL
#pragma omp parallel for collapse(3) num_threads(128) schedule(static) private(i,l,il,ir,jl,jr,kl,kr)
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l = k + j + i;
                kl = (k==0 ) ? l+NZ : l-nxny;
                kr = (k==NZ) ? l-NZ : l+nxny;
                
                dxr[l] += K * (xr[kl] + xr[kr]);
                dxi[l] += K * (xi[kl] + xi[kr]);
            }
        }
    }
    
#endif /* (TRANS_BLOCHTORREY3D_TYPE == 4) */
    
    return;
}

void trans_BlochTorreyAction4D( double *dxr, double *dxi, const double *xr, const double *xi, const double *fr, const double *fi, const double K, const double *gsize ) {
    
    const uint32_t nx       = (uint32_t)gsize[0];
    const uint32_t ny       = (uint32_t)gsize[1];
    const uint32_t nz       = (uint32_t)gsize[2];
    const uint32_t nw       = (uint32_t)gsize[3];
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
#endif /* USE_PARALLEL */
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
                    
                    dxr[l] = K * (xr[il] + xr[ir] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) + (fr[l] * xr[l] + fi[l] * xi[l]);
                    dxi[l] = K * (xi[il] + xi[ir] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) + (fr[l] * xi[l] - fi[l] * xr[l]);
                }
            }
        }
    }
    
    return;
}