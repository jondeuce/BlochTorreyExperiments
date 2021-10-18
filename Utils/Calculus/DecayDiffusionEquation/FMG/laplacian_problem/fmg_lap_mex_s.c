#include <math.h>
#include <stdint.h>
#include "mex.h"
#include <omp.h>


float ccc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float ccf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ccb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cfc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fcc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bcc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float ffc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ffb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fcf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float bbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bcb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bfb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float cfb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fcb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bcf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bfc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float occ(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ocf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ocb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ofc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float off(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ofb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float obc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float obf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float obb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float coc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float foc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float boc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float cco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cfo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float cbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ffo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float fbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bfo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float bbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float coo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float foo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float boo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float oco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ofo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float obo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float ooc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float oof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);
float oob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);

float ooo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny);


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

    const float *x = (float*)mxGetData(prhs[0]);
    const uint8_t *Mask = (const uint8_t*)mxGetData(prhs[1]);
    float h = (nrhs < 3) ? 1.0f : (float)((double*)mxGetData(prhs[2]))[0];

    const mwSize *mSize = mxGetDimensions(prhs[0]);

    plhs[0] = mxCreateNumericArray(3, mSize, mxSINGLE_CLASS, mxREAL);
    float *dx = (float*)mxGetData(plhs[0]);

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
    float(* const func_pt[])(
            const float *x, uint32_t i, uint32_t nx, uint32_t nxny) = {
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

    if(fabsf(roundf(h - 1)) > 0.0001f) {
        const float h2 = 1 / (h * h);
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



float
ccc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] - 6.0f*x[i];
}

float
fff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i+nx] + x[i+nxny]) +
           4.0f*(x[i+2] + x[i+2*nx] + x[i+2*nxny]) -
           (x[i+3] + x[i+3*nx] + x[i+3*nxny]);
}

float
bbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i-nx] + x[i-nxny]) +
           4.0f*(x[i-2] + x[i-2*nx] + x[i-2*nxny]) -
           (x[i-3] + x[i-3*nx] + x[i-3*nxny]);
}



float
ccf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] - 2.0f*x[i];
}

float
ccb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] - 2.0f*x[i];
}

float
cfc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] - 2.0f*x[i];
}

float
cbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] - 2.0f*x[i];
}

float
fcc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] - 2.0f*x[i];
}

float
bcc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] - 2.0f*x[i];
}



float
cff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

float
fcf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

float
ffc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] + 2.0f*x[i];
}

float
cbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

float
bcb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

float
bbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] + 2.0f*x[i];
}



float
ffb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i+nx] + x[i-nxny]) + 
           4.0f*(x[i+2] + x[i+2*nx] + x[i-2*nxny]) -
           (x[i+3] + x[i+3*nx] + x[i-3*nxny]);
}

float
fbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i-nx] + x[i+nxny]) +
           4.0f*(x[i+2] + x[i-2*nx] + x[i+2*nxny]) -
           (x[i+3] + x[i-3*nx] + x[i+3*nxny]);
}

float
bff(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i+nx] + x[i+nxny]) +
           4.0f*(x[i-2] + x[i+2*nx] + x[i+2*nxny]) -
           (x[i-3] + x[i+3*nx] + x[i+3*nxny]);
}

float
bbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i-nx] + x[i+nxny]) +
           4.0f*(x[i-2] + x[i-2*nx] + x[i+2*nxny]) -
           (x[i-3] + x[i-3*nx] + x[i+3*nxny]);
}

float
bfb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i-1] + x[i+nx] + x[i-nxny]) +
           4.0f*(x[i-2] + x[i+2*nx] + x[i-2*nxny]) -
           (x[i-3] + x[i+3*nx] + x[i-3*nxny]);
}

float
fbb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 6.0f*x[i] - 5.0f*(x[i+1] + x[i-nx] + x[i-nxny]) +
           4.0f*(x[i+2] + x[i-2*nx] + x[i-2*nxny]) -
           (x[i+3] + x[i-3*nx] + x[i-3*nxny]);
}



float
cfb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

float
cbf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] + x[i+1] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

float
fcb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny] + 2.0f*x[i];
}

float
bcf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] + x[i+nx] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny] + 2.0f*x[i];
}

float
fbc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3] -
           5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx] + 2.0f*x[i];
}

float
bfc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] + x[i+nxny] -
           5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3] -
           5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx] + 2.0f*x[i];
}



float
occ(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] + x[i-nxny]+x[i+nxny] - 4.0f*x[i];
}

float
ocf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

float
ocb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

float
ofc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

float
obc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

float
off(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+nx]+x[i+nxny]) + 4.0f*(x[i+2*nx]+x[i+2*nxny]) -
           x[i+3*nx]+x[i+3*nxny];
}

float
ofb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+nx]+x[i-nxny]) + 4.0f*(x[i+2*nx]+x[i-2*nxny]) -
           x[i+3*nx]+x[i-3*nxny];
}

float
obf(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-nx]+x[i+nxny]) + 4.0f*(x[i-2*nx]+x[i+2*nxny]) -
           x[i-3*nx]+x[i+3*nxny];
}

float
obb(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-nx]+x[i-nxny]) + 4.0f*(x[i-2*nx]+x[i-2*nxny]) -
           x[i-3*nx]+x[i-3*nxny];
}



float
coc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nxny]+x[i+nxny] - 4.0f*x[i];
}

float
cof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

float
cob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

float
foc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

float
fof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i+nxny]) + 4.0f*(x[i+2]+x[i+2*nxny]) -
           x[i+3]+x[i+3*nxny];
}

float
fob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i-nxny]) + 4.0f*(x[i+2]+x[i-2*nxny]) -
           x[i+3]+x[i-3*nxny];
}

float
boc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny]+x[i+nxny] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

float
bof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i+nxny]) + 4.0f*(x[i-2]+x[i+2*nxny]) -
           x[i-3]+x[i+3*nxny];
}

float
bob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i-nxny]) + 4.0f*(x[i-2]+x[i-2*nxny]) -
           x[i-3]+x[i-3*nxny];
}



float
cco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] + x[i-nx]+x[i+nx] - 4.0f*x[i];
}

float
cfo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

float
cbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1]+x[i+1] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

float
fco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

float
ffo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i+nx]) + 4.0f*(x[i+2]+x[i+2*nx]) -
           x[i+3]+x[i+3*nx];
}

float
fbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i+1]+x[i-nx]) + 4.0f*(x[i+2]+x[i-2*nx]) -
           x[i+3]+x[i-3*nx];
}

float
bco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx]+x[i+nx] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

float
bfo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i+nx]) + 4.0f*(x[i-2]+x[i+2*nx]) -
           x[i-3]+x[i+3*nx];
}

float
bbo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*(x[i-1]+x[i-nx]) + 4.0f*(x[i-2]+x[i-2*nx]) -
           x[i-3]+x[i-3*nx];
}



float
coo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-1] - 2.0f*x[i]+ x[i+1];
}

float
oco(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nx] - 2.0f*x[i]+ x[i+nx];
}

float
ooc(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return x[i-nxny] - 2.0f*x[i]+ x[i+nxny];
}

float
foo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+1] + 4.0f*x[i+2] - x[i+3];
}

float
ofo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+nx] + 4.0f*x[i+2*nx] - x[i+3*nx];
}

float
oof(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i+nxny] + 4.0f*x[i+2*nxny] - x[i+3*nxny];
}

float
boo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-1] + 4.0f*x[i-2] - x[i-3];
}

float
obo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-nx] + 4.0f*x[i-2*nx] - x[i-3*nx];
}

float
oob(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 4.0f*x[i] - 5.0f*x[i-nxny] + 4.0f*x[i-2*nxny] - x[i-3*nxny];
}

float
ooo(const float *x, uint32_t i, uint32_t nx, uint32_t nxny)
{
    return 0.0f;
}

