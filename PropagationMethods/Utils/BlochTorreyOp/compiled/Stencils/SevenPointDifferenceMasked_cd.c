#include <math.h>
#include <stdint.h>
#include <omp.h>
#include "mex.h"

/* SEVENPOINTDIFFERENCEMASKED_CD
 *
 * INPUT ARGUMENTS
 *  prhs[0] -> x:     input array (3D complex REAL array)
 *  prhs[1] -> kern:  seven point stencil (7-element REAL array)
 *  prhs[2] -> gsize: size of grid operated on (3 or 4 element REAL array)
 *  prhs[3] -> ndim:  number of dimensions operated on (scalar REAL = 3 or 4)
 *  prhs[4] -> iters: number of iterations to apply the SevenPointDifference (scalar REAL)
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

void SevenPointDifferenceCplx3D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceCplx4D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceCplxKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceCplxKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceRealKern3D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceRealKern4D( REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceReal3D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );
void SevenPointDifferenceReal4D(     REAL *dxr, REAL *dxi, const REAL *xr, const REAL *xi, const REAL *kreal, const REAL *kimag, const bool *M, const REAL *gsize );

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

    void (*SevenPointDifference)(REAL *, REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const bool *, const REAL *);
    if( ndim == 3 ) {
        if( is_kern_cplx ) {
            if( is_x_cplx ) {
                SevenPointDifference = &SevenPointDifferenceCplx3D;
            } else {
                SevenPointDifference = &SevenPointDifferenceCplxKern3D;
            }
        } else {
            if( is_x_cplx ) {
                SevenPointDifference = &SevenPointDifferenceRealKern3D;
            } else {
                SevenPointDifference = &SevenPointDifferenceReal3D;
            }
        }
    } else {
        if( is_kern_cplx ) {
            if( is_x_cplx ) {
                SevenPointDifference = &SevenPointDifferenceCplx4D;
            } else {
                SevenPointDifference = &SevenPointDifferenceCplxKern4D;
            }
        } else {
            if( is_x_cplx ) {
                SevenPointDifference = &SevenPointDifferenceRealKern4D;
            } else {
                SevenPointDifference = &SevenPointDifferenceReal4D;
            }
        }
    }

    /* Evaluate the SevenPointDifference once with input data */
    SevenPointDifference( dxr, dxi, xr, xi, kern_real, kern_imag, M, gsize );

    /* Evaluate the SevenPointDifference iters-1 times using temp variable, if necessary */
    int i;
    for(i = 1; i < iters; ++i) {
        if( is_dx_cplx ) {
            SWAP(dxr, yr, REAL*);
            SWAP(dxi, yi, REAL*);
        } else {
            SWAP(dxr, yr, REAL*);
        }
        SevenPointDifference( dxr, dxi, yr, yi, kern_real, kern_imag, M, gsize );
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
        if( is_x_cplx ) {
            mxFree(yr);
            mxFree(yi);
        } else {
            mxFree(yr);
        }
    }

    return;
}

void SevenPointDifferenceRealKern3D(
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
            jl = (j == 0  ) ? l+NY : l-nx;
            jr = (j == NY ) ? l-NY : l+nx;
            kl = (k == 0  ) ? l+NZ : l-nxny;
            kr = (k == NZ ) ? l-NZ : l+nxny;

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l+NX] - xr[l]) + mU * k_up * (xr[l+1 ] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);
            dxi[l] = k_cent * xi[l] + mD * k_down * (xi[l+NX] - xi[l]) + mU * k_up * (xi[l+1 ] - xi[l]) + mL * k_left * (xi[jl  ] - xi[l]) + mR * k_right * (xi[jr  ] - xi[l]) + mB * k_back * (xi[kl  ] - xi[l]) + mF * k_forw * (xi[kr  ] - xi[l]);

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1 ]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l-1 ] - xr[l]) + mU * k_up * (xr[l+1 ] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);
                dxi[l] = k_cent * xi[l] + mD * k_down * (xi[l-1 ] - xi[l]) + mU * k_up * (xi[l+1 ] - xi[l]) + mL * k_left * (xi[jl  ] - xi[l]) + mR * k_right * (xi[jr  ] - xi[l]) + mB * k_back * (xi[kl  ] - xi[l]) + mF * k_forw * (xi[kr  ] - xi[l]);
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1 ]; mU = M[l-NX]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

            dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l-1 ] - xr[l]) + mU * k_up * (xr[l-NX] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);
            dxi[l] = k_cent * xi[l] + mD * k_down * (xi[l-1 ] - xi[l]) + mU * k_up * (xi[l-NX] - xi[l]) + mL * k_left * (xi[jl  ] - xi[l]) + mR * k_right * (xi[jr  ] - xi[l]) + mB * k_back * (xi[kl  ] - xi[l]) + mF * k_forw * (xi[kr  ] - xi[l]);
        }
    }

    return;
}

void SevenPointDifferenceRealKern4D(
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
        SevenPointDifferenceRealKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}

void SevenPointDifferenceCplx3D(
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
            jl = ( j == 0  ) ? l+NY : l-nx;
            jr = ( j == NY ) ? l-NY : l+nx;
            kl = ( k == 0  ) ? l+NZ : l-nxny;
            kr = ( k == NZ ) ? l-NZ : l+nxny;

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l+NX] - xr[l]) + mU * kr_up * (xr[l+1 ] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l])
                 - ( ki_cent * xi[l] + mD * ki_down * (xi[l+NX] - xi[l]) + mU * ki_up * (xi[l+1 ] - xi[l]) + mL * ki_left * (xi[jl  ] - xi[l]) + mR * ki_right * (xi[jr  ] - xi[l]) + mB * ki_back * (xi[kl  ] - xi[l]) + mF * ki_forw * (xi[kr  ] - xi[l]) );
            dxi[l] = kr_cent * xi[l] + mD * kr_down * (xi[l+NX] - xi[l]) + mU * kr_up * (xi[l+1 ] - xi[l]) + mL * kr_left * (xi[jl  ] - xi[l]) + mR * kr_right * (xi[jr  ] - xi[l]) + mB * kr_back * (xi[kl  ] - xi[l]) + mF * kr_forw * (xi[kr  ] - xi[l])
                 + ( ki_cent * xr[l] + mD * ki_down * (xr[l+NX] - xr[l]) + mU * ki_up * (xr[l+1 ] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]) );

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1 ]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l-1 ] - xr[l]) + mU * kr_up * (xr[l+1 ] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l])
                     - ( ki_cent * xi[l] + mD * ki_down * (xi[l-1 ] - xi[l]) + mU * ki_up * (xi[l+1 ] - xi[l]) + mL * ki_left * (xi[jl  ] - xi[l]) + mR * ki_right * (xi[jr  ] - xi[l]) + mB * ki_back * (xi[kl  ] - xi[l]) + mF * ki_forw * (xi[kr  ] - xi[l]) );
                dxi[l] = kr_cent * xi[l] + mD * kr_down * (xi[l-1 ] - xi[l]) + mU * kr_up * (xi[l+1 ] - xi[l]) + mL * kr_left * (xi[jl  ] - xi[l]) + mR * kr_right * (xi[jr  ] - xi[l]) + mB * kr_back * (xi[kl  ] - xi[l]) + mF * kr_forw * (xi[kr  ] - xi[l])
                     + ( ki_cent * xr[l] + mD * ki_down * (xr[l-1 ] - xr[l]) + mU * ki_up * (xr[l+1 ] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]) );
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1 ]; mU = M[l-NX]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l-1 ] - xr[l]) + mU * kr_up * (xr[l-NX] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l])
                 - ( ki_cent * xi[l] + mD * ki_down * (xi[l-1 ] - xi[l]) + mU * ki_up * (xi[l-NX] - xi[l]) + mL * ki_left * (xi[jl  ] - xi[l]) + mR * ki_right * (xi[jr  ] - xi[l]) + mB * ki_back * (xi[kl  ] - xi[l]) + mF * ki_forw * (xi[kr  ] - xi[l]) );
            dxi[l] = kr_cent * xi[l] + mD * kr_down * (xi[l-1 ] - xi[l]) + mU * kr_up * (xi[l-NX] - xi[l]) + mL * kr_left * (xi[jl  ] - xi[l]) + mR * kr_right * (xi[jr  ] - xi[l]) + mB * kr_back * (xi[kl  ] - xi[l]) + mF * kr_forw * (xi[kr  ] - xi[l])
                 + ( ki_cent * xr[l] + mD * ki_down * (xr[l-1 ] - xr[l]) + mU * ki_up * (xr[l-NX] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]) );
        }
    }

    return;
}

void SevenPointDifferenceCplx4D(
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
        SevenPointDifferenceCplx3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}


void SevenPointDifferenceCplxKern3D(
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
            jl = ( j == 0  ) ? l+NY : l-nx;
            jr = ( j == NY ) ? l-NY : l+nx;
            kl = ( k == 0  ) ? l+NZ : l-nxny;
            kr = ( k == NZ ) ? l-NZ : l+nxny;

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l+NX] - xr[l]) + mU * kr_up * (xr[l+1 ] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l]);
            dxi[l] = ki_cent * xr[l] + mD * ki_down * (xr[l+NX] - xr[l]) + mU * ki_up * (xr[l+1 ] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]);

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1 ]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

                dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l-1 ] - xr[l]) + mU * kr_up * (xr[l+1 ] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l]);
                dxi[l] = ki_cent * xr[l] + mD * ki_down * (xr[l-1 ] - xr[l]) + mU * ki_up * (xr[l+1 ] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]);
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1 ]; mU = M[l-NX]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);

            dxr[l] = kr_cent * xr[l] + mD * kr_down * (xr[l-1 ] - xr[l]) + mU * kr_up * (xr[l-NX] - xr[l]) + mL * kr_left * (xr[jl  ] - xr[l]) + mR * kr_right * (xr[jr  ] - xr[l]) + mB * kr_back * (xr[kl  ] - xr[l]) + mF * kr_forw * (xr[kr  ] - xr[l]);
            dxi[l] = ki_cent * xr[l] + mD * ki_down * (xr[l-1 ] - xr[l]) + mU * ki_up * (xr[l-NX] - xr[l]) + mL * ki_left * (xr[jl  ] - xr[l]) + mR * ki_right * (xr[jr  ] - xr[l]) + mB * ki_back * (xr[kl  ] - xr[l]) + mF * ki_forw * (xr[kr  ] - xr[l]);
        }
    }

    return;
}

void SevenPointDifferenceCplxKern4D(
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
        SevenPointDifferenceCplxKern3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}


void SevenPointDifferenceReal3D(
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
            jl = ( j == 0  ) ? l+NY : l-nx;
            jr = ( j == NY ) ? l-NY : l+nx;
            kl = ( k == 0  ) ? l+NZ : l-nxny;
            kr = ( k == NZ ) ? l-NZ : l+nxny;

            bool m, mD, mU, mL, mR, mB, mF;
            REAL KD, KU, KL, KR, KB, KF;
            
            m = M[l]; mD = M[l+NX]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            /* LHS Boundary Condition */
            dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l+NX] - xr[l]) + mU * k_up * (xr[l+1 ] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);

            /* Inner Points */
            ++l, ++jl, ++jr, ++kl, ++kr;
            for(i = 1; i < nx-1; ++i) {
                m = M[l]; mD = M[l-1 ]; mU = M[l+1 ]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
                mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
                
                dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l-1 ] - xr[l]) + mU * k_up * (xr[l+1 ] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);
                ++l, ++jl, ++jr, ++kl, ++kr;
            }

            /* RHS Boundary Condition */
            m = M[l]; mD = M[l-1 ]; mU = M[l-NX]; mL = M[jl  ]; mR = M[jr  ]; mB = M[kl  ]; mF = M[kr  ];
            mD = (m == mD); mU = (m == mU); mL = (m == mL); mR = (m == mR); mB = (m == mB); mF = (m == mF);
            
            dxr[l] = k_cent * xr[l] + mD * k_down * (xr[l-1 ] - xr[l]) + mU * k_up * (xr[l-NX] - xr[l]) + mL * k_left * (xr[jl  ] - xr[l]) + mR * k_right * (xr[jr  ] - xr[l]) + mB * k_back * (xr[kl  ] - xr[l]) + mF * k_forw * (xr[kr  ] - xr[l]);
        }
    }

    return;
}

void SevenPointDifferenceReal4D(
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
        SevenPointDifferenceReal3D( &dxr[w], &dxi[w], &xr[w], &xi[w], kreal, kimag, M, gsize );
    }

    return;
}
