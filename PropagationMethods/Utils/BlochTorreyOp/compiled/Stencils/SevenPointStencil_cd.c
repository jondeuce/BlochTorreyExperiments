#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"

/* SEVENPOINTSTENCIL_CD
 *
 * INPUT ARGUMENTS
 *  prhs[0] -> x:     input array (3D complex REAL array)
 *  prhs[1] -> kern:  seven point stencil (7-element REAL array)
 *  prhs[2] -> gsize: size of grid operated on (3 or 4 element REAL array)
 *  prhs[3] -> ndim:  number of dimensions operated on (scalar REAL = 3 or 4)
 *  prhs[4] -> iters: number of iterations to apply the SevenPointStencil (scalar REAL)
 *
 * OUTPUT ARGUMENTS
 *  plhs[0] -> dx:    output array (3D complex REAL array)
 *
 */

/* Simple aliases for input pointers */
#define __x__     (prhs[0])
#define __kern__  (prhs[1])
#define __gsize__ (prhs[2])
#define __ndim__  (prhs[3])
#define __iters__ (prhs[4])

/* Simple aliases for output pointers */
#define __dx__    (plhs[0])

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

void SevenPointStencilCplx3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilCplx4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilCplxKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilCplxKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilRealKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilRealKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilReal3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );
void SevenPointStencilReal4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize );

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Seven point stencil kern */
    const REAL *kern_real = (const REAL*)mxGetData(__kern__);
    const REAL *kern_imag = (const REAL*)mxGetImagData(__kern__);

    /* Represented dimensions of input: ndim will be 3 or 4, and gsize will have respectively 3 or 4 elements */
    const REAL    *gsize = (const REAL*)mxGetData(__gsize__);
    const uint32_t ndim  = (const uint32_t) ((REAL*)mxGetData(__ndim__))[0];

    /* Number of iterations to apply */
    const uint32_t iters = (const uint32_t) ((REAL*)mxGetData(__iters__))[0];

    /* Actual dimensions of input: want to support 'flattened' 3D -> 1D, as well as full 3D */
    const mwSize *xsize = mxGetDimensions(__x__);
    const mwSize  xdim  = mxGetNumberOfDimensions(__x__);

    /* (possibly complex) input array */
    const REAL *xr = (const REAL*)mxGetData(__x__);
    const REAL *xi = (const REAL*)mxGetImagData(__x__);

    /* Flags for checking complex variables */
    const bool is_kern_cplx = mxIsComplex(__kern__);
    const bool is_x_cplx = mxIsComplex(__x__);
    const bool is_dx_cplx = is_x_cplx || is_kern_cplx;

    /* Dummy temp variable (needed for multiple iterations) */
    REAL *yr, *yi;

    if( iters > 1 ) {
        /* If we are doing multiple applications, we must make a copy to
         * store the temp. Consider e.g. y = A*(A*x) -> z = A*x0; y = A*z; */
        if( is_dx_cplx ) {
            yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
            yi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        } else {
            yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        }
    }

    /* temporary variable which will be later associated with the complex output array */
    REAL *dxr, *dxi;
    dxr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    if( is_dx_cplx ) {
        dxi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    }

    void (*SevenPointStencil)(REAL *, REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *);
    if( ndim == 3 )
        if( is_kern_cplx )
            if ( is_x_cplx )
                SevenPointStencil = &SevenPointStencilCplx3D;
            else
                SevenPointStencil = &SevenPointStencilCplxKern3D;
        else
            if ( is_x_cplx )
                SevenPointStencil = &SevenPointStencilRealKern3D;
            else
                SevenPointStencil = &SevenPointStencilReal3D;
    else
        if( is_kern_cplx )
            if ( is_x_cplx )
                SevenPointStencil = &SevenPointStencilCplx4D;
            else
                SevenPointStencil = &SevenPointStencilCplxKern4D;
        else
            if ( is_x_cplx )
                SevenPointStencil = &SevenPointStencilRealKern4D;
            else
                SevenPointStencil = &SevenPointStencilReal4D;

    /* Evaluate the SevenPointStencil once with input data */
    SevenPointStencil( dxr, dxi, xr, xi, kern_real, kern_imag, gsize );

    /* Evaluate the SevenPointStencil iters-1 times using temp variable, if necessary */
    int i;
    for(i = 1; i < iters; ++i) {
        if( is_dx_cplx ) {
            SWAP(dxr, yr, REAL*);
            SWAP(dxi, yi, REAL*);
        } else {
            SWAP(dxr, yr, REAL*);
        }
        SevenPointStencil( dxr, dxi, yr, yi, kern_real, kern_imag, gsize );
    }

    /* Create (possibly complex) output array */
    if( is_dx_cplx )
        __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxCOMPLEX); /* Create an empty array */
    else
        __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxREAL); /* Create an empty array */
    mxSetDimensions(__dx__, mxGetDimensions(__x__), mxGetNumberOfDimensions(__x__)); /* Set the dimensions to be same as input */

    /* Associate with output array */
    if( is_dx_cplx ) {
        mxSetData(__dx__, dxr); /* Assign real part */
        mxSetImagData(__dx__, dxi); /* Assign imag part */
    } else {
        mxSetData(__dx__, dxr); /* Assign real part */
    }

    /* Free temporary variable, if necessary */
    if( iters > 1 ) {
        if( is_x_cplx) {
            mxFree(yr);
            mxFree(yi);
        } else {
            mxFree(yr);
        }
    }

    return;
}

void SevenPointStencilRealKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);

    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;

    const REAL kcent = kreal[0],
            kleft = kreal[1], kright = kreal[2],
            kdown = kreal[3], kup    = kreal[4],
            kback = kreal[5], kforw  = kreal[6];

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
            dxr[l] = kcent * xr[l] + kleft * xr[l+NX] + kright * xr[l+1] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];
            dxi[l] = kcent * xi[l] + kleft * xi[l+NX] + kright * xi[l+1] + kdown * xi[jl] + kup * xi[jr] + kback * xi[kl] + kforw * xi[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                dxr[l] = kcent * xr[l] + kleft * xr[l-1] + kright * xr[l+1] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];
                dxi[l] = kcent * xi[l] + kleft * xi[l-1] + kright * xi[l+1] + kdown * xi[jl] + kup * xi[jr] + kback * xi[kl] + kforw * xi[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = kcent * xr[l] + kleft * xr[l-1] + kright * xr[l-NX] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];
            dxi[l] = kcent * xi[l] + kleft * xi[l-1] + kright * xi[l-NX] + kdown * xi[jl] + kup * xi[jr] + kback * xi[kl] + kforw * xi[kr];
        }
    }

    return;
}

void SevenPointStencilRealKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

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
        SevenPointStencilRealKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, gsize );
    }

    return;
}

void SevenPointStencilCplx3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);

    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;

    const REAL kr_cent = kreal[0],
            kr_left = kreal[1], kr_right = kreal[2],
            kr_down = kreal[3], kr_up    = kreal[4],
            kr_back = kreal[5], kr_forw  = kreal[6];

    const REAL ki_cent = kimag[0],
            ki_left = kimag[1], ki_right = kimag[2],
            ki_down = kimag[3], ki_up    = kimag[4],
            ki_back = kimag[5], ki_forw  = kimag[6];

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
            dxr[l] = kr_cent * xr[l] + kr_left * xr[l+NX] + kr_right * xr[l+1] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr]
                 - ( ki_cent * xi[l] + ki_left * xi[l+NX] + ki_right * xi[l+1] + ki_down * xi[jl] + ki_up * xi[jr] + ki_back * xi[kl] + ki_forw * xi[kr] );
            dxi[l] = kr_cent * xi[l] + kr_left * xi[l+NX] + kr_right * xi[l+1] + kr_down * xi[jl] + kr_up * xi[jr] + kr_back * xi[kl] + kr_forw * xi[kr]
                 + ( ki_cent * xr[l] + ki_left * xr[l+NX] + ki_right * xr[l+1] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr] );

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                dxr[l] = kr_cent * xr[l] + kr_left * xr[l-1] + kr_right * xr[l+1] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr]
                     - ( ki_cent * xi[l] + ki_left * xi[l-1] + ki_right * xi[l+1] + ki_down * xi[jl] + ki_up * xi[jr] + ki_back * xi[kl] + ki_forw * xi[kr] );
                dxi[l] = kr_cent * xi[l] + kr_left * xi[l-1] + kr_right * xi[l+1] + kr_down * xi[jl] + kr_up * xi[jr] + kr_back * xi[kl] + kr_forw * xi[kr]
                     + ( ki_cent * xr[l] + ki_left * xr[l-1] + ki_right * xr[l+1] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr] );
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + kr_left * xr[l-1] + kr_right * xr[l-NX] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr]
                 - ( ki_cent * xi[l] + ki_left * xi[l-1] + ki_right * xi[l-NX] + ki_down * xi[jl] + ki_up * xi[jr] + ki_back * xi[kl] + ki_forw * xi[kr] );
            dxi[l] = kr_cent * xi[l] + kr_left * xi[l-1] + kr_right * xi[l-NX] + kr_down * xi[jl] + kr_up * xi[jr] + kr_back * xi[kl] + kr_forw * xi[kr]
                 + ( ki_cent * xr[l] + ki_left * xr[l-1] + ki_right * xr[l-NX] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr] );
        }
    }

    return;
}

void SevenPointStencilCplx4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

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
        SevenPointStencilCplx3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, gsize );
    }

    return;
}


void SevenPointStencilCplxKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);

    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;

    const REAL kr_cent = kreal[0],
            kr_left = kreal[1], kr_right = kreal[2],
            kr_down = kreal[3], kr_up    = kreal[4],
            kr_back = kreal[5], kr_forw  = kreal[6];

    const REAL ki_cent = kimag[0],
            ki_left = kimag[1], ki_right = kimag[2],
            ki_down = kimag[3], ki_up    = kimag[4],
            ki_back = kimag[5], ki_forw  = kimag[6];

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
            dxr[l] = kr_cent * xr[l] + kr_left * xr[l+NX] + kr_right * xr[l+1] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
            dxi[l] = ki_cent * xr[l] + ki_left * xr[l+NX] + ki_right * xr[l+1] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                dxr[l] = kr_cent * xr[l] + kr_left * xr[l-1] + kr_right * xr[l+1] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
                dxi[l] = ki_cent * xr[l] + ki_left * xr[l-1] + ki_right * xr[l+1] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + kr_left * xr[l-1] + kr_right * xr[l-NX] + kr_down * xr[jl] + kr_up * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
            dxi[l] = ki_cent * xr[l] + ki_left * xr[l-1] + ki_right * xr[l-NX] + ki_down * xr[jl] + ki_up * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];
        }
    }

    return;
}

void SevenPointStencilCplxKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

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
        SevenPointStencilCplxKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, gsize );
    }

    return;
}


void SevenPointStencilReal3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

    const uint32_t nx     = (uint32_t)gsize[0];
    const uint32_t ny     = (uint32_t)gsize[1];
    const uint32_t nz     = (uint32_t)gsize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);

    uint32_t i, j, k, l, il, ir, jl, jr, kl, kr;

    const REAL kcent = kreal[0], kleft = kreal[1], kright = kreal[2],
                                 kdown = kreal[3], kup    = kreal[4],
                                 kback = kreal[5], kforw  = kreal[6];

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
            dxr[l] = kcent * xr[l] + kleft * xr[l+NX] + kright * xr[l+1] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                dxr[l] = kcent * xr[l] + kleft * xr[l-1] + kright * xr[l+1] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            dxr[l] = kcent * xr[l] + kleft * xr[l-1] + kright * xr[l-NX] + kdown * xr[jl] + kup * xr[jr] + kback * xr[kl] + kforw * xr[kr];
        }
    }

    return;
}

void SevenPointStencilReal4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const REAL *gsize ) {

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
        SevenPointStencilReal3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, gsize );
    }

    return;
}
