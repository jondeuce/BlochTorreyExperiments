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
 *  prhs[3] -> G:     Complex decay term Gamma = R2 + i*dw (3D complex REAL array)
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
#define SWAP(x,y,T) do { T SWAP = x; x = y; y = SWAP; } while (0)

/* Notation:
 *  x     = x[l]          = "center"
 *  xD/xU = x[l-1]/x[l+1] = "down/up"
 *  xL/xR = x[jl] /x[jr]  = "left/right"
 *  xB/xF = x[kl] /x[kr]  = "back/forw"
 */

/* Discretising using `div( D * grad(x) )` with backward divergence/forward gradient */
#define DIFFUSION_BCKDIV_D_FWDGRAD(x,xD,xU,xL,xR,xB,xF,D,DD,DL,DB) \
    ((xF + xR + xU - 3*x)*D + (xB - x)*DB + (xL - x)*DL + (xD - x)*DD)
#define DIFFUSION_BCKDIV_D_FWDGRAD_DIAG(x,xD,xU,xL,xR,xB,xF,D,DD,DL,DB) \
    ((xF + xR + xU)*D + xB*DB + xL*DL + xD*DD)

/* Discretising using `div( D * grad(x) )` with symmetrized divergence/gradient */
#define DIFFUSION_SYM_DIV_D_GRAD(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    ((xB + xF + xL + xR + xD + xU - 6*x)*D + (xB - x)*DB + (xF - x)*DF + (xL - x)*DL + (xR - x)*DR + (xD - x)*DD + (xU - x)*DU)
#define DIFFUSION_SYM_DIV_D_GRAD_DIAG(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    ((xB + xF + xL + xR + xD + xU)*D + xB*DB + xF*DF + xL*DL + xR*DR + xD*DD + xU*DU)

/* Discretising using `D * lap(x) + dot( grad(D), grad(x) )` with symmetrized gradients */
#define DIFFUSION_D_LAP_SYM_GRAD_DOT(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    (3*(xB + xF + xL + xR + xD + xU - 6*x)*D + (x - xF)*DB + (x - xB)*DF + (x - xL)*DR + (x - xR)*DL + (x - xD)*DU + (x - xU)*DD)
#define DIFFUSION_D_LAP_SYM_GRAD_DOT_DIAG(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    (3*(xB + xF + xL + xR + xD + xU)*D - xF*DB - xB*DF - xL*DR - xR*DL - xD*DU - xU*DD)

 /* Discretising using `div( D * grad(x) )` with forward/backward flux gradients with D on flux boundary */
 #define DIFFUSION_FLUXDIFF(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    ((DB + D)*xB + (DF + D)*xF + (DL + D)*xL + (DR + D)*xR + (DD + D)*xD + (DU + D)*xU - (DB + DF + DL + DR + DD + DU + 6*D)*x)
 #define DIFFUSION_FLUXDIFF_DIAG(x,xD,xU,xL,xR,xB,xF,D,DD,DU,DL,DR,DB,DF) \
    ((DB + D)*xB + (DF + D)*xF + (DL + D)*xL + (DR + D)*xR + (DD + D)*xD + (DU + D)*xU)

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
    const uint64_t ndim  = (const uint64_t) ((REAL*)mxGetData(__ndim__))[0];

    /* Number of iterations to apply */
    const uint64_t iters = (const uint64_t) ((REAL*)mxGetData(__iters__))[0];

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

    /* complex diagonal term */
    const REAL *fr = (const REAL*)mxGetData(__G__);
    const REAL *fi = (const REAL*)mxGetImagData(__G__);

    /* temporary variable which will be later associated with the complex output array */
    REAL *dxr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
    REAL *dxi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));

    void (*BTActionVariableDiff)(REAL *, REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL *, const REAL, const REAL *);
    if (ndim == 3) {
        if (isdiag) {
            BTActionVariableDiff = &BTActionVariableDiffDiagonal3D;
        } else {
            BTActionVariableDiff = &BTActionVariableDiff3D;
        }
    } else {
        if (isdiag) {
            BTActionVariableDiff = &BTActionVariableDiffDiagonal4D;
        } else {
            BTActionVariableDiff = &BTActionVariableDiff4D;
        }
    }

    /* Evaluate the BTActionVariableDiff once with input data */
    BTActionVariableDiff( dxr, dxi, xr, xi, fr, fi, Dr, Di, K, gsize );

    /* Evaluate the BTActionVariableDiff iters-1 times using temp variable, if necessary */
    REAL *yr, *yi;
    if( iters > 1 ) {
        /* If we are doing multiple applications, we must make a copy to
         * store the temp. Consider e.g. y = A*(A*x) -> z = A*x0; y = A*z; */
        yr = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        yi = mxMalloc(mxGetNumberOfElements(__x__) * sizeof(REAL));
        int i;
        for(i = 1; i < iters; ++i) {
            SWAP(dxr, yr, REAL*);
            SWAP(dxi, yi, REAL*);
            BTActionVariableDiff( dxr, dxi, yr, yi, fr, fi, Dr, Di, K, gsize );
        }
        /* Free temporary variable, if necessary */
        mxFree(yr);
        mxFree(yi);
    }

    /* Create complex output array */
    __dx__ = mxCreateNumericMatrix(0, 0, mxELEMENT_CLASS, mxCOMPLEX); /* Create an empty array */
    mxSetDimensions(__dx__, mxGetDimensions(__x__), mxGetNumberOfDimensions(__x__)); /* Set the dimensions to be same as input */

    /* Associate with output array */
    mxSetData(__dx__, dxr); /* Assign real part */
    mxSetImagData(__dx__, dxi); /* Assign imag part */

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
    const uint64_t nx     = (uint64_t)gsize[0];
    const uint64_t ny     = (uint64_t)gsize[1];
    const uint64_t nz     = (uint64_t)gsize[2];
    const uint64_t nxny   = nx*ny;
    const uint64_t nxnynz = nxny*nz;
    const uint64_t NX     = nx-1;
    const uint64_t NY     = nx*(ny-1);
    const uint64_t NZ     = nxny*(nz-1);

    uint64_t j, k;
    const uint32_t NUNROLL = 4;
    const REAL K2 = K/2;

    /* *******************************************************************
     * Triply-nested for-loop, twice collapsed
     ******************************************************************* */
#if USE_PARALLEL
#pragma omp parallel for collapse(2) OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            /* Declare index/loop variables */
            uint64_t l, i, il, ir, jl, jr, kl, kr;

            /* Periodic Boundary Conditions on y, z indexes */
            l  = j + k;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;

            /* Notation:
             *  x     = x[l]          = "center"
             *  xD/xU = x[l-1]/x[l+1] = "down/up"
             *  xL/xR = x[jl] /x[jr]  = "left/right"
             *  xB/xF = x[kl] /x[kr]  = "back/forw"
             */

            REAL    Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1, Fr1, Fi1,
                    Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2, Fr2, Fi2,
                    Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3, Fr3, Fi3,
                    Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4, Fr4, Fi4;

            /* LHS Boundary Condition */
            Xr1 = xr[l]; Xi1 = xi[l];

            XrU1 = xr[l+1]; XrU2 = xr[l+2];  XrU3 = xr[l+3];  XrU4 = xr[l+4];
            XrL1 = xr[jl];  XrL2 = xr[jl+1]; XrL3 = xr[jl+2]; XrL4 = xr[jl+3];
            XrR1 = xr[jr];  XrR2 = xr[jr+1]; XrR3 = xr[jr+2]; XrR4 = xr[jr+3];
            XrB1 = xr[kl];  XrB2 = xr[kl+1]; XrB3 = xr[kl+2]; XrB4 = xr[kl+3];
            XrF1 = xr[kr];  XrF2 = xr[kr+1]; XrF3 = xr[kr+2]; XrF4 = xr[kr+3];

            XiU1 = xi[l+1]; XiU2 = xi[l+2];  XiU3 = xi[l+3];  XiU4 = xi[l+4];
            XiL1 = xi[jl];  XiL2 = xi[jl+1]; XiL3 = xi[jl+2]; XiL4 = xi[jl+3];
            XiR1 = xi[jr];  XiR2 = xi[jr+1]; XiR3 = xi[jr+2]; XiR4 = xi[jr+3];
            XiB1 = xi[kl];  XiB2 = xi[kl+1]; XiB3 = xi[kl+2]; XiB4 = xi[kl+3];
            XiF1 = xi[kr];  XiF2 = xi[kr+1]; XiF3 = xi[kr+2]; XiF4 = xi[kr+3];

            XrD1 = xr[l+NX]; XrD2 = Xr1; Xr2 = XrU1; XrD3 = Xr2; Xr3 = XrU2; XrD4 = Xr3; Xr4 = XrU3;
            XiD1 = xi[l+NX]; XiD2 = Xi1; Xi2 = XiU1; XiD3 = Xi2; Xi3 = XiU2; XiD4 = Xi3; Xi4 = XiU3;

            // DrD1 = Dr[l+NX];
            // Dr1  = Dr[l];  Dr2  = Dr[l+1];  Dr3  = Dr[l+2];  Dr4 = Dr[l+3];
            // DrL1 = Dr[jl]; DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
            // DrB1 = Dr[kl]; DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
            // DrD2 = Dr1;    DrD3 = Dr2;      DrD4 = Dr3;

            Dr1 = Dr[l];

            DrU1 = Dr[l+1]; DrU2 = Dr[l+2];  DrU3 = Dr[l+3];  DrU4 = Dr[l+4];
            DrL1 = Dr[jl];  DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
            DrR1 = Dr[jr];  DrR2 = Dr[jr+1]; DrR3 = Dr[jr+2]; DrR4 = Dr[jr+3];
            DrB1 = Dr[kl];  DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
            DrF1 = Dr[kr];  DrF2 = Dr[kr+1]; DrF3 = Dr[kr+2]; DrF4 = Dr[kr+3];

            DrD1 = Dr[l+NX]; DrD2 = Dr1; Dr2 = DrU1; DrD3 = Dr2; Dr3 = DrU2; DrD4 = Dr3; Dr4 = DrU3;

            Fr1 = fr[l]; Fr2 = fr[l+1]; Fr3 = fr[l+2]; Fr4 = fr[l+3];
            Fi1 = fi[l]; Fi2 = fi[l+1]; Fi3 = fi[l+2]; Fi4 = fi[l+3];

            /* Discretising `div( D * grad(x) ) - Gamma * x` with finite differences */
            dxr[l]   = K2 * DIFFUSION_FLUXDIFF(Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) - (Fr1*Xr1 - Fi1*Xi1);
            dxr[l+1] = K2 * DIFFUSION_FLUXDIFF(Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) - (Fr2*Xr2 - Fi2*Xi2);
            dxr[l+2] = K2 * DIFFUSION_FLUXDIFF(Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) - (Fr3*Xr3 - Fi3*Xi3);
            dxr[l+3] = K2 * DIFFUSION_FLUXDIFF(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fr4*Xr4 - Fi4*Xi4);

            dxi[l]   = K2 * DIFFUSION_FLUXDIFF(Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) - (Fi1*Xr1 + Fr1*Xi1);
            dxi[l+1] = K2 * DIFFUSION_FLUXDIFF(Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) - (Fi2*Xr2 + Fr2*Xi2);
            dxi[l+2] = K2 * DIFFUSION_FLUXDIFF(Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) - (Fi3*Xr3 + Fr3*Xi3);
            dxi[l+3] = K2 * DIFFUSION_FLUXDIFF(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fi4*Xr4 + Fr4*Xi4);

            l+=NUNROLL; jl+=NUNROLL; jr+=NUNROLL; kl+=NUNROLL; kr+=NUNROLL;

            /* Inner Points */
            for(i = NUNROLL; i + NUNROLL < nx; i+=NUNROLL) {
                Xr1 = XrU4; Xi1 = XiU4;

                XrU1 = xr[l+1]; XrU2 = xr[l+2];  XrU3 = xr[l+3];  XrU4 = xr[l+4];
                XrL1 = xr[jl];  XrL2 = xr[jl+1]; XrL3 = xr[jl+2]; XrL4 = xr[jl+3];
                XrR1 = xr[jr];  XrR2 = xr[jr+1]; XrR3 = xr[jr+2]; XrR4 = xr[jr+3];
                XrB1 = xr[kl];  XrB2 = xr[kl+1]; XrB3 = xr[kl+2]; XrB4 = xr[kl+3];
                XrF1 = xr[kr];  XrF2 = xr[kr+1]; XrF3 = xr[kr+2]; XrF4 = xr[kr+3];

                XiU1 = xi[l+1]; XiU2 = xi[l+2];  XiU3 = xi[l+3];  XiU4 = xi[l+4];
                XiL1 = xi[jl];  XiL2 = xi[jl+1]; XiL3 = xi[jl+2]; XiL4 = xi[jl+3];
                XiR1 = xi[jr];  XiR2 = xi[jr+1]; XiR3 = xi[jr+2]; XiR4 = xi[jr+3];
                XiB1 = xi[kl];  XiB2 = xi[kl+1]; XiB3 = xi[kl+2]; XiB4 = xi[kl+3];
                XiF1 = xi[kr];  XiF2 = xi[kr+1]; XiF3 = xi[kr+2]; XiF4 = xi[kr+3];

                XrD1 = Xr4; XrD2 = Xr1; Xr2 = XrU1; XrD3 = Xr2; Xr3 = XrU2; XrD4 = Xr3; Xr4 = XrU3;
                XiD1 = Xi4; XiD2 = Xi1; Xi2 = XiU1; XiD3 = Xi2; Xi3 = XiU2; XiD4 = Xi3; Xi4 = XiU3;

                // DrD1 = Dr4;
                // Dr1  = Dr[l];  Dr2  = Dr[l+1];  Dr3  = Dr[l+2];  Dr4 = Dr[l+3];
                // DrL1 = Dr[jl]; DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
                // DrB1 = Dr[kl]; DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
                // DrD2 = Dr1;    DrD3 = Dr2;      DrD4 = Dr3;

                Dr1 = DrU4;

                DrU1 = Dr[l+1]; DrU2 = Dr[l+2];  DrU3 = Dr[l+3];  DrU4 = Dr[l+4];
                DrL1 = Dr[jl];  DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
                DrR1 = Dr[jr];  DrR2 = Dr[jr+1]; DrR3 = Dr[jr+2]; DrR4 = Dr[jr+3];
                DrB1 = Dr[kl];  DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
                DrF1 = Dr[kr];  DrF2 = Dr[kr+1]; DrF3 = Dr[kr+2]; DrF4 = Dr[kr+3];

                DrD1 = Dr4; DrD2 = Dr1; Dr2 = DrU1; DrD3 = Dr2; Dr3 = DrU2; DrD4 = Dr3; Dr4 = DrU3;

                Fr1 = fr[l]; Fr2 = fr[l+1]; Fr3 = fr[l+2]; Fr4 = fr[l+3];
                Fi1 = fi[l]; Fi2 = fi[l+1]; Fi3 = fi[l+2]; Fi4 = fi[l+3];

                /* Discretising `div( D * grad(x) ) - Gamma * x` with finite differences */
                dxr[l]   = K2 * DIFFUSION_FLUXDIFF(Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) - (Fr1*Xr1 - Fi1*Xi1);
                dxr[l+1] = K2 * DIFFUSION_FLUXDIFF(Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) - (Fr2*Xr2 - Fi2*Xi2);
                dxr[l+2] = K2 * DIFFUSION_FLUXDIFF(Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) - (Fr3*Xr3 - Fi3*Xi3);
                dxr[l+3] = K2 * DIFFUSION_FLUXDIFF(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fr4*Xr4 - Fi4*Xi4);

                dxi[l]   = K2 * DIFFUSION_FLUXDIFF(Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) - (Fi1*Xr1 + Fr1*Xi1);
                dxi[l+1] = K2 * DIFFUSION_FLUXDIFF(Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) - (Fi2*Xr2 + Fr2*Xi2);
                dxi[l+2] = K2 * DIFFUSION_FLUXDIFF(Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) - (Fi3*Xr3 + Fr3*Xi3);
                dxi[l+3] = K2 * DIFFUSION_FLUXDIFF(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fi4*Xr4 + Fr4*Xi4);

                l+=NUNROLL; jl+=NUNROLL; jr+=NUNROLL; kl+=NUNROLL; kr+=NUNROLL;
            }

            /* RHS Boundary Condition + Remainder Loop */
            for(; i < nx; ++i) {
                ir = (i==NX) ? l-NX : l+1;
                XrD4 = Xr4;  Xr4 = XrU4;  XrU4 = xr[ir]; XrL4 = xr[jl]; XrR4 = xr[jr]; XrB4 = xr[kl]; XrF4 = xr[kr];
                XiD4 = Xi4;  Xi4 = XiU4;  XiU4 = xi[ir]; XiL4 = xi[jl]; XiR4 = xi[jr]; XiB4 = xi[kl]; XiF4 = xi[kr];
                DrD4 = Dr4;  Dr4 = DrU4;  DrU4 = Dr[ir]; DrL4 = Dr[jl]; DrR4 = Dr[jr]; DrB4 = Dr[kl]; DrF4 = Dr[kr];
                // DrD4 = Dr4;  Dr4 = Dr[l]; DrL4 = Dr[jl]; DrB4 = Dr[kl];
                Fr4 = fr[l]; Fi4 = fi[l];

                /* Discretising `div( D * grad(x) ) - Gamma * x` with finite differences */
                dxr[l] = K2 * DIFFUSION_FLUXDIFF(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fr4*Xr4 - Fi4*Xi4);
                dxi[l] = K2 * DIFFUSION_FLUXDIFF(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) - (Fi4*Xr4 + Fr4*Xi4);
                ++l; ++jl; ++jr; ++kl; ++kr;
            }

            // /* LHS Boundary Condition */
            // dxr[l] = K2 * DIFFUSION_FLUXDIFF(xr[l],xr[l+NX],xr[l+1],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l+NX],Dr[jl],Dr[kl]) - (fr[l]*xr[l] - fi[l]*xi[l]);
            // dxi[l] = K2 * DIFFUSION_FLUXDIFF(xi[l],xi[l+NX],xi[l+1],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l+NX],Dr[jl],Dr[kl]) - (fi[l]*xr[l] + fr[l]*xi[l]);
            //
            // /* Inner Points */
            // ++l, ++jl, ++jr, ++kl, ++kr;
            // for(i = 1; i < nx-1; ++i) {
            //     /* Discretising `div( D * grad(x) ) - Gamma * x` with finite differences */
            //     dxr[l] = K2 * DIFFUSION_FLUXDIFF(xr[l],xr[l-1],xr[l+1],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) - (fr[l]*xr[l] - fi[l]*xi[l]);
            //     dxi[l] = K2 * DIFFUSION_FLUXDIFF(xi[l],xi[l-1],xi[l+1],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) - (fi[l]*xr[l] + fr[l]*xi[l]);
            //     ++l, ++jl, ++jr, ++kl, ++kr;
            // }
            //
            // /* RHS Boundary Condition */
            // dxr[l] = K2 * DIFFUSION_FLUXDIFF(xr[l],xr[l-1],xr[l-NX],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) - (fr[l]*xr[l] - fi[l]*xi[l]);
            // dxi[l] = K2 * DIFFUSION_FLUXDIFF(xi[l],xi[l-1],xi[l-NX],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) - (fi[l]*xr[l] + fr[l]*xi[l]);
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
    const uint64_t nx       = (uint64_t)gsize[0];
    const uint64_t ny       = (uint64_t)gsize[1];
    const uint64_t nz       = (uint64_t)gsize[2];
    const uint64_t nw       = (uint64_t)gsize[3];
    const uint64_t nxny     = nx*ny;
    const uint64_t nxnynz   = nxny*nz;
    const uint64_t nxnynznw = nxnynz*nw;
    const uint64_t NX       = nx-1;
    const uint64_t NY       = nx*(ny-1);
    const uint64_t NZ       = nxny*(nz-1);
    const uint64_t NW       = nxnynz*(nw-1);

    uint64_t w;

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
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
    const uint64_t nx     = (uint64_t)gsize[0];
    const uint64_t ny     = (uint64_t)gsize[1];
    const uint64_t nz     = (uint64_t)gsize[2];
    const uint64_t nxny   = nx*ny;
    const uint64_t nxnynz = nxny*nz;
    const uint64_t NX     = nx-1;
    const uint64_t NY     = nx*(ny-1);
    const uint64_t NZ     = nxny*(nz-1);

    uint64_t j, k;
    const uint32_t NUNROLL = 4;
    const REAL K2 = K/2;

    /* *******************************************************************
     * Triply-nested for-loop, twice collapsed
     ******************************************************************* */
#if USE_PARALLEL
#pragma omp parallel for collapse(2) OMP_PARFOR_ARGS
#endif /* USE_PARALLEL */
    for(k = 0; k < nxnynz; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            /* Declare index/loop variables */
            uint64_t l, i, il, ir, jl, jr, kl, kr;

            /* Periodic Boundary Conditions on y, z indexes */
            l  = j + k;
            jl = (j==0 ) ? l+NY : l-nx;
            jr = (j==NY) ? l-NY : l+nx;
            kl = (k==0 ) ? l+NZ : l-nxny;
            kr = (k==NZ) ? l-NZ : l+nxny;

            /* Notation:
             *  x     = x[l]          = "center"
             *  xD/xU = x[l-1]/x[l+1] = "down/up"
             *  xL/xR = x[jl] /x[jr]  = "left/right"
             *  xB/xF = x[kl] /x[kr]  = "back/forw"
             */

            REAL    Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1, Fr1, Fi1,
                    Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2, Fr2, Fi2,
                    Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3, Fr3, Fi3,
                    Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4, Fr4, Fi4;

            /* LHS Boundary Condition */
            Xr1 = xr[l]; Xi1 = xi[l];

            XrU1 = xr[l+1]; XrU2 = xr[l+2];  XrU3 = xr[l+3];  XrU4 = xr[l+4];
            XrL1 = xr[jl];  XrL2 = xr[jl+1]; XrL3 = xr[jl+2]; XrL4 = xr[jl+3];
            XrR1 = xr[jr];  XrR2 = xr[jr+1]; XrR3 = xr[jr+2]; XrR4 = xr[jr+3];
            XrB1 = xr[kl];  XrB2 = xr[kl+1]; XrB3 = xr[kl+2]; XrB4 = xr[kl+3];
            XrF1 = xr[kr];  XrF2 = xr[kr+1]; XrF3 = xr[kr+2]; XrF4 = xr[kr+3];

            XiU1 = xi[l+1]; XiU2 = xi[l+2];  XiU3 = xi[l+3];  XiU4 = xi[l+4];
            XiL1 = xi[jl];  XiL2 = xi[jl+1]; XiL3 = xi[jl+2]; XiL4 = xi[jl+3];
            XiR1 = xi[jr];  XiR2 = xi[jr+1]; XiR3 = xi[jr+2]; XiR4 = xi[jr+3];
            XiB1 = xi[kl];  XiB2 = xi[kl+1]; XiB3 = xi[kl+2]; XiB4 = xi[kl+3];
            XiF1 = xi[kr];  XiF2 = xi[kr+1]; XiF3 = xi[kr+2]; XiF4 = xi[kr+3];

            XrD1 = xr[l+NX]; XrD2 = Xr1; Xr2 = XrU1; XrD3 = Xr2; Xr3 = XrU2; XrD4 = Xr3; Xr4 = XrU3;
            XiD1 = xi[l+NX]; XiD2 = Xi1; Xi2 = XiU1; XiD3 = Xi2; Xi3 = XiU2; XiD4 = Xi3; Xi4 = XiU3;

            // DrD1 = Dr[l+NX];
            // Dr1  = Dr[l];  Dr2  = Dr[l+1];  Dr3  = Dr[l+2];  Dr4 = Dr[l+3];
            // DrL1 = Dr[jl]; DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
            // DrB1 = Dr[kl]; DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
            // DrD2 = Dr1;    DrD3 = Dr2;      DrD4 = Dr3;

            Dr1 = Dr[l];

            DrU1 = Dr[l+1]; DrU2 = Dr[l+2];  DrU3 = Dr[l+3];  DrU4 = Dr[l+4];
            DrL1 = Dr[jl];  DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
            DrR1 = Dr[jr];  DrR2 = Dr[jr+1]; DrR3 = Dr[jr+2]; DrR4 = Dr[jr+3];
            DrB1 = Dr[kl];  DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
            DrF1 = Dr[kr];  DrF2 = Dr[kr+1]; DrF3 = Dr[kr+2]; DrF4 = Dr[kr+3];

            DrD1 = Dr[l+NX]; DrD2 = Dr1; Dr2 = DrU1; DrD3 = Dr2; Dr3 = DrU2; DrD4 = Dr3; Dr4 = DrU3;

            Fr1 = fr[l]; Fr2 = fr[l+1]; Fr3 = fr[l+2]; Fr4 = fr[l+3];
            Fi1 = fi[l]; Fi2 = fi[l+1]; Fi3 = fi[l+2]; Fi4 = fi[l+3];

            /* Discretising `div( D * grad(x) ) - Diag * x` with finite differences */
            dxr[l]   = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) + (Fr1*Xr1 - Fi1*Xi1);
            dxr[l+1] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) + (Fr2*Xr2 - Fi2*Xi2);
            dxr[l+2] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) + (Fr3*Xr3 - Fi3*Xi3);
            dxr[l+3] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fr4*Xr4 - Fi4*Xi4);

            dxi[l]   = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) + (Fi1*Xr1 + Fr1*Xi1);
            dxi[l+1] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) + (Fi2*Xr2 + Fr2*Xi2);
            dxi[l+2] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) + (Fi3*Xr3 + Fr3*Xi3);
            dxi[l+3] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fi4*Xr4 + Fr4*Xi4);

            l+=NUNROLL; jl+=NUNROLL; jr+=NUNROLL; kl+=NUNROLL; kr+=NUNROLL;

            /* Inner Points */
            for(i = NUNROLL; i + NUNROLL < nx; i+=NUNROLL) {
                Xr1 = XrU4; Xi1 = XiU4;

                XrU1 = xr[l+1]; XrU2 = xr[l+2];  XrU3 = xr[l+3];  XrU4 = xr[l+4];
                XrL1 = xr[jl];  XrL2 = xr[jl+1]; XrL3 = xr[jl+2]; XrL4 = xr[jl+3];
                XrR1 = xr[jr];  XrR2 = xr[jr+1]; XrR3 = xr[jr+2]; XrR4 = xr[jr+3];
                XrB1 = xr[kl];  XrB2 = xr[kl+1]; XrB3 = xr[kl+2]; XrB4 = xr[kl+3];
                XrF1 = xr[kr];  XrF2 = xr[kr+1]; XrF3 = xr[kr+2]; XrF4 = xr[kr+3];

                XiU1 = xi[l+1]; XiU2 = xi[l+2];  XiU3 = xi[l+3];  XiU4 = xi[l+4];
                XiL1 = xi[jl];  XiL2 = xi[jl+1]; XiL3 = xi[jl+2]; XiL4 = xi[jl+3];
                XiR1 = xi[jr];  XiR2 = xi[jr+1]; XiR3 = xi[jr+2]; XiR4 = xi[jr+3];
                XiB1 = xi[kl];  XiB2 = xi[kl+1]; XiB3 = xi[kl+2]; XiB4 = xi[kl+3];
                XiF1 = xi[kr];  XiF2 = xi[kr+1]; XiF3 = xi[kr+2]; XiF4 = xi[kr+3];

                XrD1 = Xr4; XrD2 = Xr1; Xr2 = XrU1; XrD3 = Xr2; Xr3 = XrU2; XrD4 = Xr3; Xr4 = XrU3;
                XiD1 = Xi4; XiD2 = Xi1; Xi2 = XiU1; XiD3 = Xi2; Xi3 = XiU2; XiD4 = Xi3; Xi4 = XiU3;

                // DrD1 = Dr4;
                // Dr1  = Dr[l];  Dr2  = Dr[l+1];  Dr3  = Dr[l+2];  Dr4 = Dr[l+3];
                // DrL1 = Dr[jl]; DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
                // DrB1 = Dr[kl]; DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
                // DrD2 = Dr1;    DrD3 = Dr2;      DrD4 = Dr3;

                Dr1 = DrU4;

                DrU1 = Dr[l+1]; DrU2 = Dr[l+2];  DrU3 = Dr[l+3];  DrU4 = Dr[l+4];
                DrL1 = Dr[jl];  DrL2 = Dr[jl+1]; DrL3 = Dr[jl+2]; DrL4 = Dr[jl+3];
                DrR1 = Dr[jr];  DrR2 = Dr[jr+1]; DrR3 = Dr[jr+2]; DrR4 = Dr[jr+3];
                DrB1 = Dr[kl];  DrB2 = Dr[kl+1]; DrB3 = Dr[kl+2]; DrB4 = Dr[kl+3];
                DrF1 = Dr[kr];  DrF2 = Dr[kr+1]; DrF3 = Dr[kr+2]; DrF4 = Dr[kr+3];

                DrD1 = Dr4; DrD2 = Dr1; Dr2 = DrU1; DrD3 = Dr2; Dr3 = DrU2; DrD4 = Dr3; Dr4 = DrU3;

                Fr1 = fr[l]; Fr2 = fr[l+1]; Fr3 = fr[l+2]; Fr4 = fr[l+3];
                Fi1 = fi[l]; Fi2 = fi[l+1]; Fi3 = fi[l+2]; Fi4 = fi[l+3];

                /* Discretising `div( D * grad(x) ) - Diag * x` with finite differences */
                dxr[l]   = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr1, XrD1, XrU1, XrL1, XrR1, XrB1, XrF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) + (Fr1*Xr1 - Fi1*Xi1);
                dxr[l+1] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr2, XrD2, XrU2, XrL2, XrR2, XrB2, XrF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) + (Fr2*Xr2 - Fi2*Xi2);
                dxr[l+2] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr3, XrD3, XrU3, XrL3, XrR3, XrB3, XrF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) + (Fr3*Xr3 - Fi3*Xi3);
                dxr[l+3] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fr4*Xr4 - Fi4*Xi4);

                dxi[l]   = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi1, XiD1, XiU1, XiL1, XiR1, XiB1, XiF1, Dr1, DrD1, DrU1, DrL1, DrR1, DrB1, DrF1) + (Fi1*Xr1 + Fr1*Xi1);
                dxi[l+1] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi2, XiD2, XiU2, XiL2, XiR2, XiB2, XiF2, Dr2, DrD2, DrU2, DrL2, DrR2, DrB2, DrF2) + (Fi2*Xr2 + Fr2*Xi2);
                dxi[l+2] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi3, XiD3, XiU3, XiL3, XiR3, XiB3, XiF3, Dr3, DrD3, DrU3, DrL3, DrR3, DrB3, DrF3) + (Fi3*Xr3 + Fr3*Xi3);
                dxi[l+3] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fi4*Xr4 + Fr4*Xi4);

                l+=NUNROLL; jl+=NUNROLL; jr+=NUNROLL; kl+=NUNROLL; kr+=NUNROLL;
            }

            /* RHS Boundary Condition + Remainder Loop */
            for(; i < nx; ++i) {
                ir = (i==NX) ? l-NX : l+1;
                XrD4 = Xr4;  Xr4 = XrU4;  XrU4 = xr[ir]; XrL4 = xr[jl]; XrR4 = xr[jr]; XrB4 = xr[kl]; XrF4 = xr[kr];
                XiD4 = Xi4;  Xi4 = XiU4;  XiU4 = xi[ir]; XiL4 = xi[jl]; XiR4 = xi[jr]; XiB4 = xi[kl]; XiF4 = xi[kr];
                DrD4 = Dr4;  Dr4 = DrU4;  DrU4 = Dr[ir]; DrL4 = Dr[jl]; DrR4 = Dr[jr]; DrB4 = Dr[kl]; DrF4 = Dr[kr];
                // DrD4 = Dr4;  Dr4 = Dr[l]; DrL4 = Dr[jl]; DrB4 = Dr[kl];
                Fr4 = fr[l]; Fi4 = fi[l];

                /* Discretising `div( D * grad(x) ) - Diag * x` with finite differences */
                dxr[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xr4, XrD4, XrU4, XrL4, XrR4, XrB4, XrF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fr4*Xr4 - Fi4*Xi4);
                dxi[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(Xi4, XiD4, XiU4, XiL4, XiR4, XiB4, XiF4, Dr4, DrD4, DrU4, DrL4, DrR4, DrB4, DrF4) + (Fi4*Xr4 + Fr4*Xi4);
                ++l; ++jl; ++jr; ++kl; ++kr;
            }

            // /* LHS Boundary Condition */
            // dxr[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xr[l],xr[l+NX],xr[l+1],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l+NX],Dr[jl],Dr[kl]) + (fr[l]*xr[l] - fi[l]*xi[l]);
            // dxi[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xi[l],xi[l+NX],xi[l+1],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l+NX],Dr[jl],Dr[kl]) + (fi[l]*xr[l] + fr[l]*xi[l]);
            //
            // /* Inner Points */
            // ++l, ++jl, ++jr, ++kl, ++kr;
            // for(i = 1; i < nx-1; ++i) {
            //     /* Discretising `div( D * grad(x) ) - Diag * x` with finite differences */
            //     dxr[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xr[l],xr[l-1],xr[l+1],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) + (fr[l]*xr[l] - fi[l]*xi[l]);
            //     dxi[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xi[l],xi[l-1],xi[l+1],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) + (fi[l]*xr[l] + fr[l]*xi[l]);
            //     ++l, ++jl, ++jr, ++kl, ++kr;
            // }
            //
            // /* RHS Boundary Condition */
            // dxr[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xr[l],xr[l-1],xr[l-NX],xr[jl],xr[jr],xr[kl],xr[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) + (fr[l]*xr[l] - fi[l]*xi[l]);
            // dxi[l] = K2 * DIFFUSION_FLUXDIFF_DIAG(xi[l],xi[l-1],xi[l-NX],xi[jl],xi[jr],xi[kl],xi[kr],Dr[l],Dr[l-1],Dr[jl],Dr[kl]) + (fi[l]*xr[l] + fr[l]*xi[l]);
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
    const uint64_t nx       = (uint64_t)gsize[0];
    const uint64_t ny       = (uint64_t)gsize[1];
    const uint64_t nz       = (uint64_t)gsize[2];
    const uint64_t nw       = (uint64_t)gsize[3];
    const uint64_t nxny     = nx*ny;
    const uint64_t nxnynz   = nxny*nz;
    const uint64_t nxnynznw = nxnynz*nw;
    const uint64_t NX       = nx-1;
    const uint64_t NY       = nx*(ny-1);
    const uint64_t NZ       = nxny*(nz-1);
    const uint64_t NW       = nxnynz*(nw-1);

    uint64_t w;

// Parallelism should be left for the deepest nested loop, not these outer loops
// #if USE_PARALLEL
// #pragma omp parallel for OMP_PARFOR_ARGS
// #endif /* USE_PARALLEL */
    for(w = 0; w < nxnynznw; w += nxnynz) {
        BTActionVariableDiffDiagonal3D( &dxr[w], &dxi[w], &xr[w], &xi[w], fr, fi, Dr, Di, K, gsize );
    }

    return;
}
