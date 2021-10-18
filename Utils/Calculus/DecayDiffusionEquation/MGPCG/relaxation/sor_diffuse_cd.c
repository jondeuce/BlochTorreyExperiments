#include "math.h"
#include "stdint.h"
#include "mex.h"
#include "omp.h"

/* SOR_DIFFUSE_CD
 *
 * Solves Ax=b where A represents the isotropic Bloch-Torrey equation:
 *  Ax = D*lap(x) - f*x
 *
 * INPUT ARGUMENTS
 *  b:   rhs array (3D complex double array)
 *  x:   input array (3D complex double array)
 *  w:   relaxation factor (scalar double)
 *  h:   grid spacing (scalar double)
 *  D:   diffusion constant (scalar double)
 *  f:   decay term (3D complex double array)
 *  s:   arbitrary shift in decay term (scalar double)
 *  c:   multiply equation through by real constant (scalar double)
 *  it:  number of iterations (scalar uint32_t)
 *  dir: direction of smoothing (0 = forward, otherwise backward)
 */

#define USE_PARALLEL 1
#define USE_REDBLACK 1
// #define SET_NUMTHREADS num_threads(128)
#define SET_NUMTHREADS /* leave unset */

inline double icfabs2( double x, double y );
void sor_diffuse3D( 
        double *yr, double *yi, const double *xr, const double *xi,
        const double *br, const double *bi, const double *fr, const double *fi,
        const double w, const double K, const double s, const double c,
        int32_t it, const double dir, const mwSize *mSize );

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* relaxation factor w */
    const double w = (nrhs < 3) ? 1.25 : ((double*)mxGetData(prhs[2]))[0];
    
    /* grid spacing h */
    const double h = (nrhs < 4) ? 1.00 : ((double*)mxGetData(prhs[3]))[0];
    
    /* diffusion constant D */
    const double D = (nrhs < 5) ? 1.00 : ((double*)mxGetData(prhs[4]))[0];
    
    /* shift in f */
    const double s = (nrhs < 7) ? 0.00 : ((double*)mxGetData(prhs[6]))[0];
    
    /* arbitrary real constant c */
    const double c = (nrhs < 8) ? 1.00 : ((double*)mxGetData(prhs[7]))[0];
    
    /* number of iterations it */
    const int32_t it = (nrhs < 9) ? 10 : (int32_t)((double*)mxGetData(prhs[8]))[0];
	
    /* arbitrary real constant c */
    const double dir = (nrhs < 10) ? 0.0 : ((double*)mxGetData(prhs[9]))[0];
    
    /* Normalized and scaled diffusion constant K = c*D/h^2 */
    const double K = (c*D)/(h*h);
    
    /* rhs array size */
    const mwSize  ndim  = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *mSize = mxGetDimensions(prhs[0]);
    
    /* rhs array */
    const double *br = (const double*)mxGetData(prhs[0]);
    const double *bi = (const double*)mxGetImagData(prhs[0]);
    
    /* complex input array */
    const double *xr = (const double*)mxGetData(prhs[1]);
    const double *xi = (const double*)mxGetImagData(prhs[1]);
    
    /* complex decay term */
    const double *fr = (const double*)mxGetData(prhs[5]);
    const double *fi = (const double*)mxGetImagData(prhs[5]);
    
    /* complex output array */
    plhs[0] = mxCreateNumericArray(ndim, mSize, mxDOUBLE_CLASS, mxCOMPLEX);
    double *yr = (double*)mxGetData(plhs[0]);
    double *yi = (double*)mxGetImagData(plhs[0]);
    
    /* relax solution */
    sor_diffuse3D( yr, yi, xr, xi, br, bi, fr, fi, w, K, s, c, it, dir, mSize );
    
    return;

}

inline double icfabs2( double x, double y ) {
    return 1.0/(x*x + y*y);
}

#if USE_REDBLACK

void sor_diffuse3D( 
        double *yr, double *yi, const double *xr, const double *xi,
        const double *br, const double *bi, const double *fr, const double *fi,
        const double w, const double K, const double s, const double c,
        int32_t it, const double dir, const mwSize *mSize ) {
    
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    double tmp, sr, si, K2 = -6*K;
    
    double *gr, *gi;
    gr = (double*)malloc(nxnynz*sizeof(gr));
    gi = (double*)malloc(nxnynz*sizeof(gi));
    
    int32_t i, j, k, l, il, ir, jl, jr, kl, kr, iter;
    
#if USE_PARALLEL
#pragma omp parallel for private(l,tmp) SET_NUMTHREADS
#endif
    for(l = 0; l < nxnynz; ++l) {
        yr[l] = xr[l];
        yi[l] = xi[l];
        tmp   = icfabs2(K2-c*(fr[l]-s), c*fi[l]);
        gr[l] = tmp * (K2-c*(fr[l]-s));
        gi[l] = tmp * (c*fi[l]);
    }
    
    if(dir == 0.0) {
        
        /* forward smoothing */
        for(iter = 0; iter < it; ++iter) {
#if USE_PARALLEL
#pragma omp parallel for collapse(2) private(l,il,ir,jl,jr,kl,kr,sr,si) SET_NUMTHREADS
#endif
            for(k = 0; k < nxnynz; k += nxny) {
                for(j = 0; j < nxny; j += nx) {
                    l = k + j;
                    int32_t parity = l%2;
                    int32_t i0_red = parity, i0_blk = 1-parity;
                    
                    l += parity;
                    for(i = i0_red; i < nx; i+=2, l+=2) { /* RED */
                        il = (i==0 ) ? l+NX : l-1;
                        ir = (i==NX) ? l-NX : l+1;
                        jl = (j==0 ) ? l+NY : l-nx;
                        jr = (j==NY) ? l-NY : l+nx;
                        kl = (k==0 ) ? l+NZ : l-nxny;
                        kr = (k==NZ) ? l-NZ : l+nxny;
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                    
                    l = k + j + (1-parity);
                    for(i = i0_blk; i < nx; i+=2, l+=2) { /* BLACK */
                        il = (i==0 ) ? l+NX : l-1;
                        ir = (i==NX) ? l-NX : l+1;
                        jl = (j==0 ) ? l+NY : l-nx;
                        jr = (j==NY) ? l-NY : l+nx;
                        kl = (k==0 ) ? l+NZ : l-nxny;
                        kr = (k==NZ) ? l-NZ : l+nxny;
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                }
            }
            
        }
        
    } else {
        
        /* backward smoothing */
        for(iter = 0; iter < it; ++iter) {
#if USE_PARALLEL
#pragma omp parallel for collapse(2) private(l,il,ir,jl,jr,kl,kr,sr,si) SET_NUMTHREADS
#endif
            for(k = NZ; k >= 0; k -= nxny) {
                for(j = NY; j >= 0; j -= nx) {
                    l = k + j + NX;
                    int32_t parity = l%2;
                    int32_t i0_red = NX - parity, i0_blk = NX - (1-parity);
                    
                    l -= parity;
                    for(i = i0_red; i >= 0; i-=2, l-=2) { /* RED */
                        il = (i==0 ) ? l+NX : l-1;
                        ir = (i==NX) ? l-NX : l+1;
                        jl = (j==0 ) ? l+NY : l-nx;
                        jr = (j==NY) ? l-NY : l+nx;
                        kl = (k==0 ) ? l+NZ : l-nxny;
                        kr = (k==NZ) ? l-NZ : l+nxny;
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                    
                    l = k + j + NX - (1-parity);
                    for(i = i0_blk; i >= 0; i-=2, l-=2) { /* BLACK */
                        il = (i==0 ) ? l+NX : l-1;
                        ir = (i==NX) ? l-NX : l+1;
                        jl = (j==0 ) ? l+NY : l-nx;
                        jr = (j==NY) ? l-NY : l+nx;
                        kl = (k==0 ) ? l+NZ : l-nxny;
                        kr = (k==NZ) ? l-NZ : l+nxny;
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                }
            }
            
        }
        
    }

    return;
}

#else /* !USE_REDBLACK */

void sor_diffuse3D( 
        double *yr, double *yi, const double *xr, const double *xi,
        const double *br, const double *bi, const double *fr, const double *fi,
        const double w, const double K, const double s, const double c,
        int32_t it, const double dir, const mwSize *mSize ) {
    
    const uint32_t nx     = (uint32_t)mSize[0];
    const uint32_t ny     = (uint32_t)mSize[1];
    const uint32_t nz     = (uint32_t)mSize[2];
    const uint32_t nxny   = nx*ny;
    const uint32_t nxnynz = nxny*nz;
    const uint32_t NX     = nx-1;
    const uint32_t NY     = nx*(ny-1);
    const uint32_t NZ     = nxny*(nz-1);
    
    double tmp, sr, si, K2 = -6*K;
    
    double *gr, *gi;
    gr = (double*)malloc(nxnynz*sizeof(gr));
    gi = (double*)malloc(nxnynz*sizeof(gi));
    
    int32_t i, j, k, l, il, ir, jl, jr, kl, kr, iter;
    
#if USE_PARALLEL
#pragma omp parallel for private(l,tmp) SET_NUMTHREADS
#endif
    for(l = 0; l < nxnynz; ++l) {
        yr[l] = xr[l];
        yi[l] = xi[l];
        tmp   = icfabs2(K2-c*(fr[l]-s), c*fi[l]);
        gr[l] = tmp * (K2-c*(fr[l]-s));
        gi[l] = tmp * (c*fi[l]);
    }
    
    if(dir == 0.0) {
        
        /* forward smoothing */
        for(iter = 0; iter < it; ++iter) {
            
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
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                }
            }
            
        }
        
    } else {
        
        /* backward smoothing */
        for(iter = 0; iter < it; ++iter) {
            
            for(k = NZ; k >= 0; k -= nxny) {
                for(j = NY; j >= 0; j -= nx) {
                    l = k + j + NX;
                    for(i = NX; i >= 0; --i, --l) {
                        il = (i==0 ) ? l+NX : l-1;
                        ir = (i==NX) ? l-NX : l+1;
                        jl = (j==0 ) ? l+NY : l-nx;
                        jr = (j==NY) ? l-NY : l+nx;
                        kl = (k==0 ) ? l+NZ : l-nxny;
                        kr = (k==NZ) ? l-NZ : l+nxny;
                        
                        sr = K * (yr[il] + yr[ir] + yr[jl] + yr[jr] + yr[kl] + yr[kr]);
                        si = K * (yi[il] + yi[ir] + yi[jl] + yi[jr] + yi[kl] + yi[kr]);
                        yr[l] += w * ( gr[l] * (br[l]-sr) - gi[l] * (bi[l]-si) - yr[l] );
                        yi[l] += w * ( gr[l] * (bi[l]-si) + gi[l] * (br[l]-sr) - yi[l] );
                    }
                    
                }
            }
            
        }
        
    }

    return;
}

#endif /* USE_REDBLACK */