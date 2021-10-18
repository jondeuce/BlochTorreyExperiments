#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>


#define BOO 1
#define FOO 2
#define COO 3
#define CCC 4


void lapf(float *dx,
        const float *x, const uint8_t *Mask, double *h, const mwSize *siz);
void lapd(double *dx,
        const double *x, const uint8_t *Mask, double *h, const mwSize *siz);

float cf(const float *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
float ff(const float *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
float bf(const float *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
float of(const float *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);

double cd(const double *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
double fd(const double *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
double bd(const double *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);
double od(const double *x,
        uint32_t l, uint32_t a, uint32_t N, uint32_t i, const uint8_t *Mask);


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs < 2)
        mexErrMsgTxt("Phase and Mask needed.");
    
    if(!mxIsLogical(prhs[1]))
        mexErrMsgTxt("Mask must be logical.");
    
    const uint8_t is_s = mxIsSingle(prhs[0]);
    const uint8_t is_d = mxIsDouble(prhs[0]);
    
    if(!is_s && !is_d)
        mexErrMsgTxt("Phase must be single or double.");
    
    const mwSize *siz = mxGetDimensions(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[1]);
    
    double *h = (double[3]){1.0, 1.0, 1.0};
    if(nrhs > 2) {
        const double *hh = (const double*)mxGetData(prhs[2]);
        h = mxGetNumberOfElements(prhs[2]) < 3
            ? (double[3]){1.0/hh[0], 1.0/hh[0], 1.0/hh[0]}
            : (double[3]){1.0/hh[0], 1.0/hh[1], 1.0/hh[2]};
            
        h[0] = h[0]*h[0];
        h[1] = h[1]*h[1];
        h[2] = h[2]*h[2];
    }
    
    if(is_s) {
        const float *x = (const float*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxSINGLE_CLASS, mxREAL);
        float *dx = (float*)mxGetData(plhs[0]);
        
        lapf(dx, x, Mask, h, siz);
    } else {
        const double *x = (const double*)mxGetData(prhs[0]);
        plhs[0] = mxCreateNumericArray(3, siz, mxDOUBLE_CLASS, mxREAL);
        double *dx = (double*)mxGetData(plhs[0]);
        
        lapd(dx, x, Mask, h, siz);
    }
    return;
}


void lapf(float *dx,
        const float *x, const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;
    uint8_t idx;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t N    = nxny*nz;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    const uint32_t NZZ  = NZ-nxny;
    const uint32_t NYY  = NY-nx;
    const uint32_t NXX  = NX-1;
    
    const float hx2 = (float)h[0];
    const float hy2 = (float)h[1];
    const float hz2 = (float)h[2];
    
    /*
     *  2*Mask[i+a] + Mask[i-a]:
     *      0 -> (o)
     *      1 -> (b)ackward
     *      2 -> (f)orward
     *      3 -> (c)entral
     */
    float(* const func_pt[])(const float *x, uint32_t l, uint32_t a,
            uint32_t N, uint32_t i, const uint8_t *Mask) = {
        of, bf, ff, cf
    };
        
	#pragma omp parallel for private(l, idx) collapse(3) schedule(dynamic, 32768)
    for(k = 0; k < N; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l  = k + j + i;
                if(Mask[l]) {
                    idx = (i-1) < NXX
                        ? (Mask[l+1] << 1) + Mask[l-1]
                        : i == NX ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, 1, nx, i, Mask)*hy2;
                            
                    idx = (j-nx) < NYY
                        ? (Mask[l+nx] << 1) + Mask[l-nx]
                        : j == NY ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, nx, nxny, j, Mask)*hx2;
                     
                    idx = (k-nxny) < NZZ
                        ? (Mask[l+nxny] << 1) + Mask[l-nxny]
                        : k == NZ ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, nxny, N, k, Mask)*hz2;
                }
            }
        }
    }
    return;
}


void lapd(double *dx,
        const double *x, const uint8_t *Mask, double *h, const mwSize *siz)
{
    uint32_t i, j, k, l;
    uint8_t idx;

    const uint32_t nx   = (uint32_t)siz[0];
    const uint32_t ny   = (uint32_t)siz[1];
    const uint32_t nz   = (uint32_t)siz[2];
    const uint32_t nxny = nx*ny;
    const uint32_t N    = nxny*nz;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;
    const uint32_t NZZ  = NZ-nxny;
    const uint32_t NYY  = NY-nx;
    const uint32_t NXX  = NX-1;
    
    const double hx2 = h[0];
    const double hy2 = h[1];
    const double hz2 = h[2];
    
    /*
     *  2*Mask[i+a] + Mask[i-a]:
     *      0 -> (o)
     *      1 -> (b)ackward
     *      2 -> (f)orward
     *      3 -> (c)entral
     */
    double(* const func_pt[])(const double *x, uint32_t l, uint32_t a,
            uint32_t N, uint32_t i, const uint8_t *Mask) = {
//         od, bd, fd, cd
//         od, cd, cd, cd
        od, od, od, cd
    };
    
	#pragma omp parallel for private(l, idx) collapse(3) schedule(dynamic, 32768)
    for(k = 0; k < N; k += nxny) {
        for(j = 0; j < nxny; j += nx) {
            for(i = 0; i < nx; ++i) {
                l  = k + j + i;
                if(Mask[l]) {
                    idx = (i-1) < NXX
                        ? (Mask[l+1] << 1) + Mask[l-1]
                        : i == NX ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, 1, nx, i, Mask)*hy2;
                    
                    idx = (j-nx) < NYY
                        ? (Mask[l+nx] << 1) + Mask[l-nx]
                        : j == NY ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, nx, nxny, j, Mask)*hx2;
                     
                    idx = (k-nxny) < NZZ
                        ? (Mask[l+nxny] << 1) + Mask[l-nxny]
                        : k == NZ ? BOO : FOO;

                    dx[l] += (*func_pt[idx])(x, l, nxny, N, k, Mask)*hz2;
                }
            }
        }
    }
    return;
}


float
cf(const float *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    return x[l-a] + x[l+a] - 2.0f*x[l];
}


float
ff(const float *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    if((i+3*a < N) && (Mask[l+3*a]) && (Mask[l+2*a]))
            return 2.0f*x[l] - 5.0f*x[l+a] + 4.0f*x[l+2*a] - x[l+3*a];
    
    if((i+2*a < N) && (Mask[l+2*a]))
            return x[l] - 2.0f*x[l+a] + x[l+2*a];
    
    return 0.0f;
}


float
bf(const float *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    if((i-3*a < N) && (Mask[l-3*a]) && (Mask[l-2*a]))
            return 2.0f*x[l] - 5.0f*x[l-a] + 4.0f*x[l-2*a] - x[l-3*a];
    
    if((i-2*a < N) && (Mask[l-2*a]))
            return x[l] - 2.0f*x[l-a] + x[l-2*a];
    
    return 0.0f;
}


float
of(const float *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    return 0.0f;
}


double
cd(const double *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    return x[l-a] + x[l+a] - 2.0*x[l];
}


double
fd(const double *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    if((i+3*a < N) && (Mask[l+3*a]) && (Mask[l+2*a]))
            return 2.0*x[l] - 5.0*x[l+a] + 4.0*x[l+2*a] - x[l+3*a];
    
    if((i+2*a < N) && (Mask[l+2*a]))
            return x[l] - 2.0*x[l+a] + x[l+2*a];
    
    return 0.0;
}


double
bd(const double *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    if((i-3*a < N) && (Mask[l-3*a]) && (Mask[l-2*a]))
            return 2.0*x[l] - 5.0*x[l-a] + 4.0*x[l-2*a] - x[l-3*a];
    
    if((i-2*a < N) && (Mask[l-2*a]))
            return x[l] - 2.0*x[l-a] + x[l-2*a];
    
    return 0.0;
}


double
od(const double *x, uint32_t l, uint32_t a, uint32_t N, uint32_t i,
        const uint8_t *Mask)
{
    return 0.0;
}
