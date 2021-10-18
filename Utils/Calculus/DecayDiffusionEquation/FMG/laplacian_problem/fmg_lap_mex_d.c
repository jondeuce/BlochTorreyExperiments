#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>


double ccc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double ccf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ccb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cfc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fcc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bcc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double ffc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ffb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fcf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double bbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bcb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bfb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double cfb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fcb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bcf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bfc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double occ(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ocf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ocb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ofc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double off(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ofb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double obc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double obf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double obb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double coc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double foc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double boc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double cco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cfo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double cbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ffo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double fbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bfo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double bbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double coo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double foo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double boo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double oco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ofo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double obo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double ooc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double oof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);
double oob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);

double ooo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny);


#define OPTS 4

#define BACK 1
#define FORW 2

#define JX_0 FORW
#define JY_0 FORW*OPTS
#define JZ_0 FORW*OPTS*OPTS

#define JX_N BACK
#define JY_N BACK*OPTS
#define JZ_N BACK*OPTS*OPTS

#define FLAG OPTS+1


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    uint32_t i, j, k, l;
    uint8_t jx, jy, jz, by, bz;

    const double *x = (double*)mxGetData(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[1]);
    double h = (nrhs < 3) ? 1.0 : ((double*)mxGetData(prhs[2]))[0];

    const mwSize *mSize = mxGetDimensions(prhs[0]);

    plhs[0] = mxCreateNumericArray(3, mSize, mxDOUBLE_CLASS, mxREAL);
    double *dx = (double*)mxGetData(plhs[0]);

    const uint32_t nx   = (uint32_t)mSize[0];
    const uint32_t ny   = (uint32_t)mSize[1];
    const uint32_t nz   = (uint32_t)mSize[2];
    const uint32_t nxny = nx*ny;
    const uint32_t N    = nxny*nz;
    const uint32_t NZ   = nxny*(nz-1);
    const uint32_t NY   = nx*(ny-1);
    const uint32_t NX   = nx-1;

    /*
     *  2*Mask[i+a] + Mask[i-a]:
     *      0 -> (o)
     *      1 -> (b)ackward
     *      2 -> (f)orward
     *      3 -> (c)entral
     */
    double(* const func_pt[])(
            const double *x, uint32_t i, uint32_t nx, uint32_t nxny) = {
        ooo, boo, foo, coo, obo, bbo, fbo, cbo,
        ofo, bfo, ffo, cfo, oco, bco, fco, cco,
        oob, bob, fob, cob, obb, bbb, fbb, cbb,
        ofb, bfb, ffb, cfb, ocb, bcb, fcb, ccb,
        oof, bof, fof, cof, obf, bbf, fbf, cbf,
        off, bff, fff, cff, ocf, bcf, fcf, ccf,
        ooc, boc, foc, coc, obc, bbc, fbc, cbc,
        ofc, bfc, ffc, cfc, occ, bcc, fcc, ccc
    };

    uint8_t *ind;
    ind = mxCalloc(N, sizeof(*ind));

    for(k = 0; k < N; k += nxny) {
        bz = k == 0 ? JZ_0 : (k == NZ ? JZ_N : FLAG);

        for(j = 0; j < nxny; j += nx) {
            by = j == 0 ? JY_0 : (j == NY ? JY_N : FLAG);
            l  = k + j;

            for(i = 0; i < nx; ++i, ++l) {
                if(Mask[l]) {
                    jz = bz != FLAG
                         ? bz
                         : (Mask[l+nxny] << 5) + (Mask[l-nxny] << 4);
                    jy = by != FLAG
                         ? by
                         : (Mask[l+nx] << 3) + (Mask[l-nx] << 2);
                    jx = i == 0 ? JX_0 : i == NX
                         ? JX_N
                         : (Mask[l+1] << 1) + (Mask[l-1]);
                    ind[l] = (jx + jy + jz);
                }
            }
        }
    }

    if(fabs(round(h - 1)) > 0.0001) {
        const double h2 = 1 / (h * h);
        #pragma omp parallel for
        for(l = 0; l < N; ++l) {
            dx[l] = (*func_pt[ind[l]])(x, l, nx, nxny) * h2;
        }
    } else {
        #pragma omp parallel for
        for(l = 0; l < N; ++l) {
            dx[l] = (*func_pt[ind[l]])(x, l, nx, nxny);
        }
    }

    mxFree(ind);
    return;

}



double
ccc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] - 6.0f*x[i];
}

double
fff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i+nx] + x[i+nxny]) +
           4.0f*(x[i+2] + x[i+2*nx] + x[i+2*nxny]) -
           (x[i+3] + x[i+3*nx] + x[i+3*nxny]);
}

double
bbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i-nx] + x[i-nxny]) +
           4.0f*(x[i-2] + x[i-2*nx] + x[i-2*nxny]) -
           (x[i-3] + x[i-3*nx] + x[i-3*nxny]);
}



double
ccf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] - 2.0f*x[i];
}

double
ccb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] - 2.0f*x[i];
}

double
cfc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] - 2.0f*x[i];
}

double
cbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] - 2.0f*x[i];
}

double
fcc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] - 2.0f*x[i];
}

double
bcc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] - 2.0f*x[i];
}



double
cff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

double
fcf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

double
ffc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] + 2.0f*x[i];
}

double
cbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

double
bcb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

double
bbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] + 2.0f*x[i];
}



double
ffb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i+nx] + x[i-nxny]) + 
           4.0f*(x[i+2] + x[i+2*nx] + x[i-2*nxny]) -
           (x[i+3] + x[i+3*nx] + x[i-3*nxny]);
}

double
fbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i-nx] + x[i+nxny]) +
           4.0f*(x[i+2] + x[i-2*nx] + x[i+2*nxny]) -
           (x[i+3] + x[i-3*nx] + x[i+3*nxny]);
}

double
bff(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i+nx] + x[i+nxny]) +
           4.0f*(x[i-2] + x[i+2*nx] + x[i+2*nxny]) -
           (x[i-3] + x[i+3*nx] + x[i+3*nxny]);
}

double
bbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i-nx] + x[i+nxny]) +
           4.0f*(x[i-2] + x[i-2*nx] + x[i+2*nxny]) -
           (x[i-3] + x[i-3*nx] + x[i+3*nxny]);
}

double
bfb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i+nx] + x[i-nxny]) +
           4.0f*(x[i-2] + x[i+2*nx] + x[i-2*nxny]) -
           (x[i-3] + x[i+3*nx] + x[i-3*nxny]);
}

double
fbb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i-nx] + x[i-nxny]) +
           4.0f*(x[i+2] + x[i-2*nx] + x[i-2*nxny]) -
           (x[i+3] + x[i-3*nx] + x[i-3*nxny]);
}



double
cfb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

double
cbf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

double
fcb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

double
bcf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

double
fbc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] + 2.0f*x[i];
}

double
bfc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] + 2.0f*x[i];
}



double
occ(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] - 4.0f*x[i];
}

double
ocf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

double
ocb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

double
ofc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

double
obc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

double
off(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+nx]+x[i+nxny]) + 4.0f*(x[i+2*nx]+x[i+2*nxny]) -
           x[i+3*nx]+x[i+3*nxny];
}

double
ofb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+nx]+x[i-nxny]) + 4.0f*(x[i+2*nx]+x[i-2*nxny]) -
           x[i+3*nx]+x[i-3*nxny];
}

double
obf(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-nx]+x[i+nxny]) + 4.0f*(x[i-2*nx]+x[i+2*nxny]) -
           x[i-3*nx]+x[i+3*nxny];
}

double
obb(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-nx]+x[i-nxny]) + 4.0f*(x[i-2*nx]+x[i-2*nxny]) -
           x[i-3*nx]+x[i-3*nxny];
}



double
coc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] - 4.0f*x[i];
}

double
cof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

double
cob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

double
foc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

double
fof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i+nxny]) + 4.0f*(x[i+2]+x[i+2*nxny]) -
           x[i+3]+x[i+3*nxny];
}

double
fob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i-nxny]) + 4.0f*(x[i+2]+x[i-2*nxny]) -
           x[i+3]+x[i-3*nxny];
}

double
boc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

double
bof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i+nxny]) + 4.0f*(x[i-2]+x[i+2*nxny]) -
           x[i-3]+x[i+3*nxny];
}

double
bob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i-nxny]) + 4.0f*(x[i-2]+x[i-2*nxny]) -
           x[i-3]+x[i-3*nxny];
}



double
cco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] - 4.0f*x[i];
}

double
cfo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

double
cbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

double
fco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

double
ffo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i+nx]) + 4.0f*(x[i+2]+x[i+2*nx]) -
           x[i+3]+x[i+3*nx];
}

double
fbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i-nx]) + 4.0f*(x[i+2]+x[i-2*nx]) -
           x[i+3]+x[i-3*nx];
}

double
bco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

double
bfo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i+nx]) + 4.0f*(x[i-2]+x[i+2*nx]) -
           x[i-3]+x[i+3*nx];
}

double
bbo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i-nx]) + 4.0f*(x[i-2]+x[i-2*nx]) -
           x[i-3]+x[i-3*nx];
}



double
coo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] - 2.0f*x[i]+ x[i+1];
}

double
oco(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] - 2.0f*x[i]+ x[i+nx];
}

double
ooc(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] - 2.0f*x[i]+ x[i+nxny];
}

double
foo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

double
ofo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

double
oof(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

double
boo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

double
obo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

double
oob(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

double
ooo(const double *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 0.0f;
}

