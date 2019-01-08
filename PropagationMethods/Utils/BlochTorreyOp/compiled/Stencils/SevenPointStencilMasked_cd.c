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
#define __M__     (prhs[5])

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

void SevenPointStencilCplx3D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilCplx4D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilCplxKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilCplxKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilRealKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilRealKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilReal3D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointStencilReal4D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );

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

    /* boolean mask */
    const bool *M  = (const bool*)mxGetLogicals(__M__);

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

    void (*SevenPointStencil)(REAL *, REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const bool *, const REAL *);
    if (ndim == 3) {
        if (is_kern_cplx) {
            if (is_x_cplx) {
                SevenPointStencil = &SevenPointStencilCplx3D;
            } else {
                SevenPointStencil = &SevenPointStencilCplxKern3D;
            }
        } else {
            if (is_x_cplx) {
                SevenPointStencil = &SevenPointStencilRealKern3D;
            } else {
                SevenPointStencil = &SevenPointStencilReal3D;
            }
        }
    } else {
        if (is_kern_cplx) {
            if (is_x_cplx) {
                SevenPointStencil = &SevenPointStencilCplx4D;
            } else {
                SevenPointStencil = &SevenPointStencilCplxKern4D;
            }
        } else {
            if (is_x_cplx) {
                SevenPointStencil = &SevenPointStencilRealKern4D;
            } else {
                SevenPointStencil = &SevenPointStencilReal4D;
            }
        }
    }

    /* Evaluate the SevenPointStencil once with input data */
    SevenPointStencil( dxr, dxi, xr, xi, kern_real, kern_imag, M, gsize );

    /* Evaluate the SevenPointStencil iters-1 times using temp variable, if necessary */
    int i;
    for(i = 1; i < iters; ++i) {
        if( is_dx_cplx ) {
            SWAP(dxr, yr, REAL*);
            SWAP(dxi, yi, REAL*);
        } else {
            SWAP(dxr, yr, REAL*);
        }
        SevenPointStencil( dxr, dxi, yr, yi, kern_real, kern_imag, M, gsize );
    }

    /* Create (possibly complex) output array */
    if( is_dx_cplx ) {
        __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxCOMPLEX); /* Create an empty array */
    } else {
        __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxREAL); /* Create an empty array */
    }
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

void SevenPointStencilRealKern3D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

    const REAL k_cent = kreal[0], k_down = kreal[1], k_up    = kreal[2],
                                  k_left = kreal[3], k_right = kreal[4],
                                  k_back = kreal[5], k_forw  = kreal[6];

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

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = k_cent * xr[l] + mD * k_down * xr[l+NX] + mU * k_up * xr[l+1] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];
            dxi[l] = k_cent * xi[l] + mD * k_down * xi[l+NX] + mU * k_up * xi[l+1] + mL * k_left * xi[jl] + mR * k_right * xi[jr] + mB * k_back * xi[kl] + mF * k_forw * xi[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = k_cent * xr[l] + mD * k_down * xr[l-1] + mU * k_up * xr[l+1] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];
                dxi[l] = k_cent * xi[l] + mD * k_down * xi[l-1] + mU * k_up * xi[l+1] + mL * k_left * xi[jl] + mR * k_right * xi[jr] + mB * k_back * xi[kl] + mF * k_forw * xi[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1]; mU = M[l-NX]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

            dxr[l] = k_cent * xr[l] + mD * k_down * xr[l-1] + mU * k_up * xr[l-NX] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];
            dxi[l] = k_cent * xi[l] + mD * k_down * xi[l-1] + mU * k_up * xi[l-NX] + mL * k_left * xi[jl] + mR * k_right * xi[jr] + mB * k_back * xi[kl] + mF * k_forw * xi[kr];
        }
    }

    return;
}

void SevenPointStencilRealKern4D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        SevenPointStencilRealKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}

void SevenPointStencilCplx3D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

    const REAL kr_cent = kreal[0],
            kr_down = kreal[1], kr_up    = kreal[2],
            kr_left = kreal[3], kr_right = kreal[4],
            kr_back = kreal[5], kr_forw  = kreal[6];

    const REAL ki_cent = kimag[0],
            ki_down = kimag[1], ki_up    = kimag[2],
            ki_left = kimag[3], ki_right = kimag[4],
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

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + mD * kr_down * xr[l+NX] + mU * kr_up * xr[l+1] + mL * kr_left * xr[jl] + mR * kr_right * xr[jr] + mB * kr_back * xr[kl] + mF * kr_forw * xr[kr]
                 - ( ki_cent * xi[l] + mD * ki_down * xi[l+NX] + mU * ki_up * xi[l+1] + mL * ki_left * xi[jl] + mR * ki_right * xi[jr] + mB * ki_back * xi[kl] + mF * ki_forw * xi[kr] );
            dxi[l] = kr_cent * xi[l] + mD * kr_down * xi[l+NX] + mU * kr_up * xi[l+1] + mL * kr_left * xi[jl] + mR * kr_right * xi[jr] + mB * kr_back * xi[kl] + mF * kr_forw * xi[kr]
                 + ( ki_cent * xr[l] + mD * ki_down * xr[l+NX] + mU * ki_up * xr[l+1] + mL * ki_left * xr[jl] + mR * ki_right * xr[jr] + mB * ki_back * xr[kl] + mF * ki_forw * xr[kr] );

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = kr_cent * xr[l] + mD * kr_down * xr[l-1] + mU * kr_up * xr[l+1] + mL * kr_left * xr[jl] + mR * kr_right * xr[jr] + mB * kr_back * xr[kl] + mF * kr_forw * xr[kr]
                     - ( ki_cent * xi[l] + mD * ki_down * xi[l-1] + mU * ki_up * xi[l+1] + mL * ki_left * xi[jl] + mR * ki_right * xi[jr] + mB * ki_back * xi[kl] + mF * ki_forw * xi[kr] );
                dxi[l] = kr_cent * xi[l] + mD * kr_down * xi[l-1] + mU * kr_up * xi[l+1] + mL * kr_left * xi[jl] + mR * kr_right * xi[jr] + mB * kr_back * xi[kl] + mF * kr_forw * xi[kr]
                     + ( ki_cent * xr[l] + mD * ki_down * xr[l-1] + mU * ki_up * xr[l+1] + mL * ki_left * xr[jl] + mR * ki_right * xr[jr] + mB * ki_back * xr[kl] + mF * ki_forw * xr[kr] );
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1]; mU = M[l-NX]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            dxr[l] = kr_cent * xr[l] + mD * kr_down * xr[l-1] + mU * kr_up * xr[l-NX] + mL * kr_left * xr[jl] + mR * kr_right * xr[jr] + mB * kr_back * xr[kl] + mF * kr_forw * xr[kr]
                 - ( ki_cent * xi[l] + mD * ki_down * xi[l-1] + mU * ki_up * xi[l-NX] + mL * ki_left * xi[jl] + mR * ki_right * xi[jr] + mB * ki_back * xi[kl] + mF * ki_forw * xi[kr] );
            dxi[l] = kr_cent * xi[l] + mD * kr_down * xi[l-1] + mU * kr_up * xi[l-NX] + mL * kr_left * xi[jl] + mR * kr_right * xi[jr] + mB * kr_back * xi[kl] + mF * kr_forw * xi[kr]
                 + ( ki_cent * xr[l] + mD * ki_down * xr[l-1] + mU * ki_up * xr[l-NX] + mL * ki_left * xr[jl] + mR * ki_right * xr[jr] + mB * ki_back * xr[kl] + mF * ki_forw * xr[kr] );
        }
    }

    return;
}

void SevenPointStencilCplx4D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        SevenPointStencilCplx3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}


void SevenPointStencilCplxKern3D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

    const REAL kr_cent = kreal[0],
            kr_down = kreal[1], kr_up    = kreal[2],
            kr_left = kreal[3], kr_right = kreal[4],
            kr_back = kreal[5], kr_forw  = kreal[6];

    const REAL ki_cent = kimag[0],
            ki_down = kimag[1], ki_up    = kimag[2],
            ki_left = kimag[3], ki_right = kimag[4],
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

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + kr_down * xr[l+NX] + kr_up * xr[l+1] + kr_left * xr[jl] + kr_right * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
            dxi[l] = ki_cent * xr[l] + ki_down * xr[l+NX] + ki_up * xr[l+1] + ki_left * xr[jl] + ki_right * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = kr_cent * xr[l] + kr_down * xr[l-1] + kr_up * xr[l+1] + kr_left * xr[jl] + kr_right * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
                dxi[l] = ki_cent * xr[l] + ki_down * xr[l-1] + ki_up * xr[l+1] + ki_left * xr[jl] + ki_right * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1]; mU = M[l-NX]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

            dxr[l] = kr_cent * xr[l] + kr_down * xr[l-1] + kr_up * xr[l-NX] + kr_left * xr[jl] + kr_right * xr[jr] + kr_back * xr[kl] + kr_forw * xr[kr];
            dxi[l] = ki_cent * xr[l] + ki_down * xr[l-1] + ki_up * xr[l-NX] + ki_left * xr[jl] + ki_right * xr[jr] + ki_back * xr[kl] + ki_forw * xr[kr];
        }
    }

    return;
}

void SevenPointStencilCplxKern4D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        SevenPointStencilCplxKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}


void SevenPointStencilReal3D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

    const REAL k_cent = kreal[0], k_down = kreal[1], k_up    = kreal[2],
                                  k_left = kreal[3], k_right = kreal[4],
                                  k_back = kreal[5], k_forw  = kreal[6];

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

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = k_cent * xr[l] + mD * k_down * xr[l+NX] + mU * k_up * xr[l+1] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1]; mU = M[l+1]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
                
                dxr[l] = k_cent * xr[l] + mD * k_down * xr[l-1] + mU * k_up * xr[l+1] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1]; mU = M[l-NX]; mL = M[jl]; mR = M[jr]; mB = M[kl]; mF = M[kr];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            dxr[l] = k_cent * xr[l] + mD * k_down * xr[l-1] + mU * k_up * xr[l-NX] + mL * k_left * xr[jl] + mR * k_right * xr[jr] + mB * k_back * xr[kl] + mF * k_forw * xr[kr];
        }
    }

    return;
}

void SevenPointStencilReal4D(
    REAL *dxr, REAL *dxi,
    const REAL *xr, const REAL *xi,
    const REAL *kreal, const REAL *kimag,
    const bool *M,
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

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        SevenPointStencilReal3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}
