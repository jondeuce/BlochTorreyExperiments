#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"

/* LAPLACIAN_CD
 *
 * INPUT ARGUMENTS
 *  prhs[0] -> x:     input array (3D complex REAL array)
 *  prhs[1] -> h:     grid spacing (scalar REAL)
 *  prhs[2] -> gsize: size of grid operated on (3 or 4 element REAL array)
 *  prhs[3] -> ndim:  number of dimensions operated on (scalar REAL = 3 or 4)
 *  prhs[4] -> iters: number of iterations to apply the Laplacian (scalar REAL)
 *
 * OUTPUT ARGUMENTS
 *  plhs[0] -> dx:    output array (3D complex REAL array)
 *
 */

/* Simple aliases for input pointers */
#define __x__ (prhs[0])
#define __h__ (prhs[1])
#define __gsize__ (prhs[2])
#define __ndim__ (prhs[3])
#define __iters__ (prhs[4])

/* Simple aliases for output pointers */
#define __dx__ (plhs[0])

/* Defines for omp parallel for loops */
#define USE_PARALLEL 1
// #define NUMTHREADS 128
#define NUM_THREADS (omp_get_max_threads())
// #define OMP_PARFOR_ARGS
#define OMP_PARFOR_ARGS schedule(static) num_threads(NUM_THREADS)

/* Alias for basic element type, for easier switching between single/double */
#define REALTYPE 1 /* 1 for double, 0 for single */
#if REALTYPE
#define REAL double
#define mxELEMENT_CLASS mxDOUBLE_CLASS
#else
#define REAL float
#define mxELEMENT_CLASS mxSINGLE_CLASS
#endif /* REALTYPE */

/* Simple unsafe swap macro: https://stackoverflow.com/questions/3982348/implement-generic-swap-macro-in-c */
#define SWAP(x,y,T) do { T SWAP = x; x = y; y = SWAP; } while (0)

void Laplacian3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL K, const REAL *gsize );
void Laplacian4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL K, const REAL *gsize );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* grid spacing h */
    const REAL h = ((const REAL*)mxGetData(__h__))[0];
        
    /* Normalized and scaled diffusion constant K = D/h^2 */
    const REAL K = 1.0/(h*h);
    
    /* Represented dimensions of input: ndim will be 3 or 4, and gsize will have respectively 3 or 4 elements */
    const REAL    *gsize = (const REAL*)mxGetData(__gsize__);
    const uint32_t ndim  = (const uint32_t) ((REAL*)mxGetData(__ndim__))[0];
    
    /* Number of iterations to apply */
    const uint32_t iters = (const uint32_t) ((REAL*)mxGetData(__iters__))[0];
    
    /* Actual dimensions of input: want to support 'flattened' 3D -> 1D, as well as full 3D */
    const mwSize *xsize = mxGetDimensions(__x__);
    const mwSize  xdim  = mxGetNumberOfDimensions(__x__);
    
    /* complex input array */
    const REAL *xr = (const REAL*)mxGetData(__x__);
    const REAL *xi = (const REAL*)mxGetImagData(__x__);
    
    /* Dummy temp variable (needed for multiple iterations) */
    REAL *yr, *yi;
    
    if( iters > 1 ) {
        /* If we are doing multiple applications, we must make a copy to
         * store the temp. Consider e.g. y = A*(A*x) -> z = A*x0; y = A*z; */
        yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        yi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    }
    
    /* temporary variable which will be later associated with the complex output array */
    REAL *dxr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    REAL *dxi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    
    void (*Laplacian)(REAL *, REAL *, const REAL *, const REAL *, const REAL, const REAL *);
    if( ndim == 3 )
        Laplacian = &Laplacian3D;
    else
        Laplacian = &Laplacian4D;
    
    /* Evaluate the Laplacian once with input data */
    Laplacian( dxr, dxi, xr, xi, K, gsize );
    
    /* Evaluate the Laplacian iters-1 times using temp variable, if necessary */
    int i;
    for(i = 1; i < iters; ++i) {
        SWAP(dxr, yr, REAL*);
        SWAP(dxi, yi, REAL*);
        Laplacian( dxr, dxi, yr, yi, K, gsize );
    }
    
    /* Create complex output array */
    __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxCOMPLEX); /* Create an empty array */
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

void Laplacian3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL K, const REAL *gsize ) {
    
    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;
    REAL K6 = 6*K;
    
#if USE_PARALLEL
#pragma omp parallel for collapse(2) OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            /* Periodic Boundary Conditions on y, z indexes */
            l = k + j;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;
            
            /* LHS Boundary Condition */
            dxr[l] = K * (xr[l+NX] + xr[l+1] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - K6 * xr[l];
            dxi[l] = K * (xi[l+NX] + xi[l+1] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - K6 * xi[l];
            
            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                dxr[l] = K * (xr[l-1] + xr[l+1] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - K6 * xr[l];
                dxi[l] = K * (xi[l-1] + xi[l+1] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - K6 * xi[l];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }
            
            /* RHS Boundary Condition */
            dxr[l] = K * (xr[l-1] + xr[l-NX] + xr[jl] + xr[jr] + xr[kl] + xr[kr]) - K6 * xr[l];
            dxi[l] = K * (xi[l-1] + xi[l-NX] + xi[jl] + xi[jr] + xi[kl] + xi[kr]) - K6 * xi[l];
        }
    }
    
    return;
}

void Laplacian4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL K, const REAL *gsize ) {
    
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
        
    int64_t w = 0;
    
#if USE_PARALLEL
#pragma omp parallel for OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        Laplacian3D( &dxr[w], &dxi[w], &xr[w], &xi[w], K, gsize );
    }
    
    return;
}