#include <stdio.h>
#include <string.h>
#include <math.h>

#define BLOCKS 512
#define THREADS_PER_BLOCK 1024 

//ldoc on
/**
 * ## Implementation
 *
 * The actually work of computing the fluxes and speeds is done
 * by local (`static`) helper functions that take as arguments
 * pointers to all the individual fields.  This is helpful to the
 * compilers, since by specifying the `restrict` keyword, we are
 * promising that we will not access the field data through the
 * wrong pointer.  This lets the compiler do a better job with
 * vectorization.
 */

extern "C" {
#include "shallow2d.h"
}

static const float g = 9.8;

static
void shallow2dv_flux(float* __restrict__ fh,
                     float* __restrict__ fhu,
                     float* __restrict__ fhv,
                     float* __restrict__ gh,
                     float* __restrict__ ghu,
                     float* __restrict__ ghv,
                     const float* __restrict__ h,
                     const float* __restrict__ hu,
                     const float* __restrict__ hv,
                     float g,
                     int ncell)
{
    memcpy(fh, hu, ncell * sizeof(float));
    memcpy(gh, hv, ncell * sizeof(float));
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i], hui = hu[i], hvi = hv[i];
        float inv_h = 1/hi;
        fhu[i] = hui*hui*inv_h + (0.5f*g)*hi*hi;
        fhv[i] = hui*hvi*inv_h;
        ghu[i] = hui*hvi*inv_h;
        ghv[i] = hvi*hvi*inv_h + (0.5f*g)*hi*hi;
    }
}


static
void shallow2dv_speed(float* __restrict__ cxy,
                      const float* __restrict__ h,
                      const float* __restrict__ hu,
                      const float* __restrict__ hv,
                      float g,
                      int ncell)
{
    float cx = cxy[0];
    float cy = cxy[1];
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i];
        float inv_hi = 1.0f/h[i];
        float root_gh = sqrtf(g * hi);
        float cxi = fabsf(hu[i] * inv_hi) + root_gh;
        float cyi = fabsf(hv[i] * inv_hi) + root_gh;
        if (cx < cxi) cx = cxi;
        if (cy < cyi) cy = cyi;
    }
    cxy[0] = cx;
    cxy[1] = cy;
}

void shallow2d_flux(float* FU, float* GU, const float* U,
                    int ncell, int field_stride)
{
    shallow2dv_flux(FU, FU+field_stride, FU+2*field_stride,
                    GU, GU+field_stride, GU+2*field_stride,
                    U,  U +field_stride, U +2*field_stride,
                    g, ncell);
}

/*
void shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride)
{
    shallow2dv_speed(cxy, U, U+field_stride, U+2*field_stride, g, ncell);
}
*/
/**
 *  Compute the maximum of 2 single-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the minimum
 * @param[in]	value	The value that is compared to the reference in order to determine the minimum
 */

__device__
void AtomicMax(float * const address, const float value)
{
    if (* address >= value) { return; }

    int * const address_as_i = (int *)address;
    int old = * address_as_i, assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) { break; }

        old = atomicCAS(address_as_i, assumed, __float_as_int(value));
        } while (assumed != old);
}

__global__
void cuda_speed (float* cx,
                 float* cy,
                 const float* h,
                 const float* hu,
                 const float* hv)
{
  __shared__ float tempx[THREADS_PER_BLOCK];
  __shared__ float tempy[THREADS_PER_BLOCK];
  static const float g = 9.8;
  int index = threadIdx.x;

  float hi = h[index];
  float inv_hi = 1.0f/h[index];
  float root_gh = sqrtf(g * hi);
  tempx[index] = fabsf(hu[index] * inv_hi) + root_gh;
  tempy[index] = fabsf(hv[index] * inv_hi) + root_gh;

  __syncthreads();

  if ( 0 == threadIdx.x ) {
    float x, y = 0.0;
    for (int i = 0 ; i < THREADS_PER_BLOCK; i++)  {
      x = fmaxf(x, tempx[i]);
      y = fmaxf(y, tempy[i]);
    }
    // *cx = fmaxf(*cx,x);
    // *cy = fmaxf(*cy,y);
    AtomicMax(cx, x);
    AtomicMax(cy, y);
  }
}

void shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride)
{
    int size = 3 * ncell * sizeof(float);
    float *cuda_U, *cx, *cy;
    
    cudaMalloc( (void**)&cuda_U, size );
    cudaMalloc( (void**)&cx, sizeof(float) );
    cudaMalloc( (void**)&cy, sizeof(float) );

    cudaMemcpy( cx, &cxy[0], sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( cy, &cxy[1], sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( cuda_U, U, size, cudaMemcpyHostToDevice );

    //shallow2dv_speed(cxy, U, U+field_stride, U+2*field_stride, g, ncell);
    cuda_speed<<<BLOCKS, THREADS_PER_BLOCK>>>(cx, cy, cuda_U, cuda_U+field_stride, cuda_U+2*field_stride);
    cudaMemcpy(&cxy[0], cx, sizeof(float),  cudaMemcpyDeviceToHost);
    cudaMemcpy(&cxy[1], cy, sizeof(float),  cudaMemcpyDeviceToHost);

    cudaFree(cuda_U);
    cudaFree(cx);
    cudaFree(cy);
}
