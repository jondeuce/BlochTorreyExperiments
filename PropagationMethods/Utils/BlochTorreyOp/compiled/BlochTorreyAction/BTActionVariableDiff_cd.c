#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"

/* BTACTIONVARIABLEDIFF_CD
 *
 * INPUT ARGUMENTS
 *  prhs[0] -> x:     input array (3D complex REAL array)
 *  prhs[1] -> h:     grid spacing (scalar REAL)
 *  prhs[2] -> D:     diffusion constant (3D REAL array)
 *  prhs[3] -> f:     Complex decay term Gamma = R2 + i*dw (3D complex REAL array)
 *  prhs[4] -> gsize: size of grid operated on (3 or 4 element REAL array)
 *  prhs[5] -> ndim:  number of dimensions operated on (scalar REAL = 3 or 4)
 *  prhs[6] -> iters: number of iterations to apply the BTActionVariableDiff (scalar REAL)
 *
 * OUTPUT ARGUMENTS
 *  plhs[0] -> dx:    output array (3D complex REAL array)
 *
 */

/* Simple aliases for input pointers */
#define __x__ (prhs[0])
#define __h__ (prhs[1])
#define __D__ (prhs[2])
#define __G__ (prhs[3])
#define __gsize__ (prhs[4])
#define __ndim__ (prhs[5])
#define __iters__ (prhs[6])
#define __isdiag__ (prhs[7])

/* Simple aliases for output pointers */
#define __dx__ (plhs[0])

/* Defines for omp parallel for loops */
#define USE_PARALLEL 1
// #define NUMTHREADS 128
#define NUM_THREADS (omp_get_max_threads())
// #define OMP_PARFOR_ARGS
#define OMP_PARFOR_ARGS schedule(static) num_threads(NUM_THREADS)

/* Enum for different implementations (Type 1 currently fastest) */
#define BLOCHTORREY3D_TYPE 1

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
#define SWAP(x,y) do { typeof(x) SWAP = x; x = y; y = SWAP; } while (0)

void BTActionVariableDiff3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *fr, const REAL *fi, const REAL *Dr, const REAL *Di, const REAL K, const REAL *gsize );
void BTActionVariableDiff4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *fr, const REAL *fi, const REAL *Dr, const REAL *Di, const REAL K, const REAL *gsize );
void BTActionVariableDiffDiagonal3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *fr, const REAL *fi, const REAL *Dr, const REAL *Di, const REAL K, const REAL *gsize );
void BTActionVariableDiffDiagonal4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *fr, const REAL *fi, const REAL *Dr, const REAL *Di, const REAL K, const REAL *gsize );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* grid spacing h */
    const REAL h = ((const REAL*)mxGetData(__h__))[0];
    
    /* Inverse square grid size constant K = 1/h^2 */
    const REAL K = 1.0/(h*h);
    
    /* Represented dimensions of input: ndim will be 3 or 4, and gsize will have respectively 3 or 4 elements */
    const REAL  *gsize = (const REAL*)mxGetData(__gsize__);
    const uint32_t ndim  = (const uint32_t) ((REAL*)mxGetData(__ndim__))[0];
    
    /* Number of iterations to apply */
    const uint32_t iters = (const uint32_t) ((REAL*)mxGetData(__iters__))[0];
    
    /* Flag for diagonal vs. Gamma input */
    const bool isdiag = mxIsLogicalScalarTrue(__isdiag__);
    
    /* Actual dimensions of input: want to support 'flattened' 3D -> 1D, as well as full 3D */
    const mwSize *xsize = mxGetDimensions(__x__);
    const mwSize  xdim  = mxGetNumberOfDimensions(__x__);
    
    /* complex input array */
    const REAL *xr = (const REAL*)mxGetData(__x__);
    const REAL *xi = (const REAL*)mxGetImagData(__x__);
    
    /* Diffusion coefficient input array */
    const REAL *Dr = (const REAL*)mxGetData(__D__);
    const REAL *Di = (const REAL*)mxGetImagData(__D__);
    
    /* Dummy temp variable (needed for multiple iterations) */
    REAL *yr, *yi;
    
    if( iters > 1 ) {
        /* If we are doing multiple applications, we must make a copy to
         * store the temp. Consider e.g. y = A*(A*x) -> z = A*x0; y = A*z; */
        yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        yi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    }
    
    /* complex diagonal term */
    const REAL *fr = (const REAL*)mxGetData(__G__);
    const REAL *fi = (const REAL*)mxGetImagData(__G__);
    
    /* temporary variable which will be later associated with the complex output array */
    REAL *dxr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    REAL *dxi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    
    void (*BTActionVariableDiff)(REAL *, REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL, const REAL *);
    if( ndim == 3 )
        if( isdiag )
            BTActionVariableDiff = &BTActionVariableDiffDiagonal3D;
        else
            BTActionVariableDiff = &BTActionVariableDiff3D;
    else
        if( isdiag )
            BTActionVariableDiff = &BTActionVariableDiffDiagonal4D;
        else
            BTActionVariableDiff = &BTActionVariableDiff4D;
    
    /* Evaluate the BTActionVariableDiff once with input data */
    BTActionVariableDiff( dxr, dxi, xr, xi, fr, fi, Dr, Di, K, gsize );
    
    /* Evaluate the BTActionVariableDiff iters-1 times using temp variable, if necessary */
    int i;
    for(i = 1; i < iters; ++i) {
        SWAP(dxr, yr);
        SWAP(dxi, yi);
        BTActionVariableDiff( dxr, dxi, yr, yi, fr, fi, Dr, Di, K, gsize );
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

/* *******************************************************************
 * Bloch-Torrey action when the input is Gamma
 ******************************************************************* */
void BTActionVariableDiff3D(
        REAL *dxr, REAL *dxi,
        const REAL *xr, const REAL *xi,
        const REAL *fr, const REAL *fi,
        const REAL *Dr, const REAL *Di,
        const REAL K,
        const REAL *gsize
        )
{
    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;
    const REAL Khalf = 0.5 * K;
    
    /* *******************************************************************
     * Triply-nested for-loop, twice collapsed
     ******************************************************************* */
#if USE_PARALLEL
#pragma omp parallel for collapse(2) OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            /* Periodic Boundary Conditions on y, z indexes */
            l  = j + k;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;

            /* LHS Boundary Condition */
            dxr[l] = K * ((xr[kr] + xr[jr] + xr[l+1] - 3*xr[l])*Dr[l] + (xr[kl] - xr[l])*Dr[kl] + (xr[jl] - xr[l])*Dr[jl] + (xr[l+NX] - xr[l])*Dr[l+NX])
                       - (fr[l]*xr[l] - fi[l]*xi[l]);
            dxi[l] = K * ((xi[kr] + xi[jr] + xi[l+1] - 3*xi[l])*Dr[l] + (xi[kl] - xi[l])*Dr[kl] + (xi[jl] - xi[l])*Dr[jl] + (xi[l+NX] - xi[l])*Dr[l+NX])
                       - (fi[l]*xr[l] + fr[l]*xi[l]);

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                /* Discretising using `div( D * grad(x) )` with backward divergence/forward gradient */
                dxr[l] = K * ((xr[kr] + xr[jr] + xr[l+1] - 3*xr[l])*Dr[l] + (xr[kl] - xr[l])*Dr[kl] + (xr[jl] - xr[l])*Dr[jl] + (xr[l-1] - xr[l])*Dr[l-1])
                           - (fr[l]*xr[l] - fi[l]*xi[l]);
                dxi[l] = K * ((xi[kr] + xi[jr] + xi[l+1] - 3*xi[l])*Dr[l] + (xi[kl] - xi[l])*Dr[kl] + (xi[jl] - xi[l])*Dr[jl] + (xi[l-1] - xi[l])*Dr[l-1])
                           - (fi[l]*xr[l] + fr[l]*xi[l]);

                /*
                 * Discretising using `div( D * grad(x) )` with symmetrized divergence/gradient
                dxr[l] = Khalf * ((xr[kl] - xr[l])*Dr[kl] + (xr[kl] + xr[kr] + xr[jl] + xr[jr] + xr[l-1] + xr[l+1] - 6*xr[l])*Dr[l] + (xr[kr] - xr[l])*Dr[kr] + (xr[jl] - xr[l])*Dr[jl] + (xr[jr] - xr[l])*Dr[jr] + (xr[l-1] - xr[l])*Dr[l-1] + (xr[l+1] - xr[l])*Dr[l+1])
                               - (fr[l]*xr[l] - fi[l]*xi[l]);
                dxi[l] = Khalf * ((xi[kl] - xi[l])*Dr[kl] + (xi[kl] + xi[kr] + xi[jl] + xi[jr] + xi[l-1] + xi[l+1] - 6*xi[l])*Dr[l] + (xi[kr] - xi[l])*Dr[kr] + (xi[jl] - xi[l])*Dr[jl] + (xi[jr] - xi[l])*Dr[jr] + (xi[l-1] - xi[l])*Dr[l-1] + (xi[l+1] - xi[l])*Dr[l+1])
                               - (fi[l]*xr[l] + fr[l]*xi[l]);
                 */

                /*
                 * Discretising using `D * lap(x) + dot( grad(D), grad(x) )` with symmetrized gradients
                dxr[l] = Khalf * ((xr[l] - xr[kr])*Dr[kl] + (xr[l] - xr[kl])*Dr[kr] + (xr[l] - xr[jl])*Dr[jr] + (xr[l] - xr[jr])*Dr[jl] + (xr[l] - xr[l-1])*Dr[l+1] + (xr[l] - xr[l+1])*Dr[l-1] + 3*(xr[kl] + xr[kr] + xr[jl] + xr[jr] + xr[l-1] + xr[l+1] - 6*xr[l])*Dr[l])
                               - (fr[l]*xr[l] - fi[l]*xi[l]);
                dxi[l] = Khalf * ((xi[l] - xi[kr])*Dr[kl] + (xi[l] - xi[kl])*Dr[kr] + (xi[l] - xi[jl])*Dr[jr] + (xi[l] - xi[jr])*Dr[jl] + (xi[l] - xi[l-1])*Dr[l+1] + (xi[l] - xi[l+1])*Dr[l-1] + 3*(xi[kl] + xi[kr] + xi[jl] + xi[jr] + xi[l-1] + xi[l+1] - 6*xi[l])*Dr[l])
                               - (fi[l]*xr[l] + fr[l]*xi[l]);
                 */
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = K * ((xr[kr] + xr[jr] + xr[l-NX] - 3*xr[l])*Dr[l] + (xr[kl] - xr[l])*Dr[kl] + (xr[jl] - xr[l])*Dr[jl] + (xr[l-1] - xr[l])*Dr[l-1])
                       - (fr[l]*xr[l] - fi[l]*xi[l]);
            dxi[l] = K * ((xi[kr] + xi[jr] + xi[l-NX] - 3*xi[l])*Dr[l] + (xi[kl] - xi[l])*Dr[kl] + (xi[jl] - xi[l])*Dr[jl] + (xi[l-1] - xi[l])*Dr[l-1])
                       - (fi[l]*xr[l] + fr[l]*xi[l]);
        }
    }
    
    return;
}

void BTActionVariableDiff4D(
        REAL *dxr, REAL *dxi,
        const REAL *xr, const REAL *xi,
        const REAL *fr, const REAL *fi,
        const REAL *Dr, const REAL *Di,
        const REAL K,
        const REAL *gsize
        )
{
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
        BTActionVariableDiff3D( &dxr[w], &dxi[w], &xr[w], &xi[w], fr, fi, Dr, Di, K, gsize );
    }
        
    return;
}


/* *******************************************************************
 * Bloch-Torrey action when the input is the matrix diagonal
 ******************************************************************* */
void BTActionVariableDiffDiagonal3D(
        REAL *dxr, REAL *dxi,
        const REAL *xr, const REAL *xi,
        const REAL *fr, const REAL *fi,
        const REAL *Dr, const REAL *Di,
        const REAL K,
        const REAL *gsize
        )
{
    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;
    const REAL Khalf = 0.5 * K;
    
    /* *******************************************************************
     * Triply-nested for-loop, twice collapsed
     ******************************************************************* */
#if USE_PARALLEL
#pragma omp parallel for collapse(2) OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            /* Periodic Boundary Conditions on y, z indexes */
            l  = j + k;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;

            /* LHS Boundary Condition */
            dxr[l] = K * ((xr[kr] + xr[jr] + xr[l+1])*Dr[l] + xr[kl]*Dr[kl] + xr[jl]*Dr[jl] + xr[l+NX]*Dr[l+NX])
                       + (fr[l]*xr[l] - fi[l]*xi[l]);
            dxi[l] = K * ((xi[kr] + xi[jr] + xi[l+1])*Dr[l] + xi[kl]*Dr[kl] + xi[jl]*Dr[jl] + xi[l+NX]*Dr[l+NX])
                       + (fi[l]*xr[l] + fr[l]*xi[l]);

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                /* Discretising using `div( D * grad(x) )` with backward divergence/forward gradient */
                dxr[l] = K * ((xr[kr] + xr[jr] + xr[l+1])*Dr[l] + xr[kl]*Dr[kl] + xr[jl]*Dr[jl] + xr[l-1]*Dr[l-1])
                           + (fr[l]*xr[l] - fi[l]*xi[l]);
                dxi[l] = K * ((xi[kr] + xi[jr] + xi[l+1])*Dr[l] + xi[kl]*Dr[kl] + xi[jl]*Dr[jl] + xi[l-1]*Dr[l-1])
                           + (fi[l]*xr[l] + fr[l]*xi[l]);

                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = K * ((xr[kr] + xr[jr] + xr[l-NX])*Dr[l] + xr[kl]*Dr[kl] + xr[jl]*Dr[jl] + xr[l-1]*Dr[l-1])
                       + (fr[l]*xr[l] - fi[l]*xi[l]);
            dxi[l] = K * ((xi[kr] + xi[jr] + xi[l-NX])*Dr[l] + xi[kl]*Dr[kl] + xi[jl]*Dr[jl] + xi[l-1]*Dr[l-1])
                       + (fi[l]*xr[l] + fr[l]*xi[l]);
        }
    }
    
    return;
}

void BTActionVariableDiffDiagonal4D(
        REAL *dxr, REAL *dxi,
        const REAL *xr, const REAL *xi,
        const REAL *fr, const REAL *fi,
        const REAL *Dr, const REAL *Di,
        const REAL K,
        const REAL *gsize
        )
{
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
        BTActionVariableDiffDiagonal3D( &dxr[w], &dxi[w], &xr[w], &xi[w], fr, fi, Dr, Di, K, gsize );
    }
        
    return;
}