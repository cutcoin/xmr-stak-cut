#include "xmrstak/backend/cryptonight.hpp"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
    if (waitTime > 0)
    {
        if (waitTime > 100)
        {
            // use a waitable timer for larger intervals > 0.1ms

            HANDLE timer;
            LARGE_INTEGER ft;

            ft.QuadPart = -10ll * int64_t(waitTime); // Convert to 100 nanosecond interval, negative value indicates relative time

            timer = CreateWaitableTimer(NULL, TRUE, NULL);
            SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        }
        else
        {
            // use a polling loop for short intervals <= 100ms

            LARGE_INTEGER perfCnt, start, now;
            __int64 elapsed;

            QueryPerformanceFrequency(&perfCnt);
            QueryPerformanceCounter(&start);
            do {
		SwitchToThread();
                QueryPerformanceCounter((LARGE_INTEGER*) &now);
                elapsed = (__int64)((now.QuadPart - start.QuadPart) / (float)perfCnt.QuadPart * 1000 * 1000);
            } while ( elapsed < waitTime );
        }
    }
}
#else
#include <unistd.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
	usleep(waitTime);
}
#endif

#include "cryptonight.hpp"
#include "cuda_extra.hpp"
#include "cuda_aes.hpp"
#include "cuda_device.hpp"

/* sm_2X is limited to 2GB due to the small TLB
 * therefore we never use 64bit indices
 */
#if defined(XMR_STAK_LARGEGRID) && (__CUDA_ARCH__ >= 300)
typedef uint64_t IndexType;
#else
typedef int IndexType;
#endif

__device__ __forceinline__ uint64_t cuda_mul128( uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi )
{
	*product_hi = __umul64hi( multiplier, multiplicand );
	return (multiplier * multiplicand );
}

template< typename T >
__device__ __forceinline__ T loadGlobal64( T * const addr )
{
	T x;
	asm volatile( "ld.global.cg.u64 %0, [%1];" : "=l"( x ) : "l"( addr ) );
	return x;
}

template< typename T >
__device__ __forceinline__ T loadGlobal32( T * const addr )
{
	T x;
	asm volatile( "ld.global.cg.u32 %0, [%1];" : "=r"( x ) : "l"( addr ) );
	return x;
}


template< typename T >
__device__ __forceinline__ void storeGlobal32( T* addr, T const & val )
{
	asm volatile( "st.global.cg.u32 [%0], %1;" : : "l"( addr ), "r"( val ) );
}

template<size_t ITERATIONS, uint32_t THREAD_SHIFT>
__global__ void cryptonight_core_gpu_phase1( int threads, int bfactor, int partidx, uint32_t * __restrict__ long_state, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
	const int sub = ( threadIdx.x & 7 ) << 2;

	const int batchsize = ITERATIONS >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40], text[4];

	MEMCPY8( key, ctx_key1 + thread * 40, 20 );

	if( partidx == 0 )
	{
		// first round
		MEMCPY8( text, ctx_state + thread * 50 + sub + 16, 2 );
	}
	else
	{
		// load previous text data
		MEMCPY8( text, &long_state[( (uint64_t) thread << THREAD_SHIFT ) + sub + start - 32], 2 );
	}
	__syncthreads( );
	for ( int i = start; i < end; i += 32 )
	{
		cn_aes_pseudo_round_mut( sharedMemory, text, key );
		MEMCPY8(&long_state[((uint64_t) thread << THREAD_SHIFT) + (sub + i)], text, 2);
	}
}

/** avoid warning `unused parameter` */
template< typename T >
__forceinline__ __device__ void unusedVar( const T& )
{
}

/** shuffle data for
 *
 * - this method can be used with all compute architectures
 * - for <sm_30 shared memory is needed
 *
 * @param ptr pointer to shared memory, size must be `threadIdx.x * sizeof(uint32_t)`
 *            value can be NULL for compute architecture >=sm_30
 * @param sub thread number within the group, range [0;4)
 * @param value value to share with other threads within the group
 * @param src thread number within the group from where the data is read, range [0;4)
 */
__forceinline__ __device__ uint32_t shuffle(volatile uint32_t* ptr,const uint32_t sub,const int val,const uint32_t src)
{
#if( __CUDA_ARCH__ < 300 )
    ptr[sub] = val;
    return ptr[src&3];
#else
    unusedVar( ptr );
    unusedVar( sub );
#   if(__CUDACC_VER_MAJOR__ >= 9)
    return __shfl_sync(0xFFFFFFFF, val, src, 4 );
#	else
	return __shfl( val, src, 4 );
#	endif
#endif
}

template<size_t ITERATIONS, uint32_t THREAD_SHIFT, uint32_t MASK>
#ifdef XMR_STAK_THREADS
__launch_bounds__( XMR_STAK_THREADS * 4 )
#endif
__global__ void cryptonight_core_gpu_phase2( int threads, int bfactor, int partidx, uint32_t * d_long_state, uint32_t * d_ctx_a, uint32_t * d_ctx_b )
{
	__shared__ uint32_t sharedMemory[1024];
  extern __shared__ uint32_t shared_a[];
  uint32_t *shared_b = shared_a + blockDim.x;
  uint32_t *shared_c = shared_b + blockDim.x;

	cn_aes_gpu_init( sharedMemory );

	__syncthreads( );

	const int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 2;
	const int sub = threadIdx.x & 3;

  if ( thread >= threads )
		return;

	int i;
  uint32_t al, bl, cl, tmpl, idx;
	const int batchsize = (ITERATIONS * 2) >> ( 1 + bfactor );
	const int start = partidx * batchsize;
	const int end = start + batchsize;
	uint32_t *long_state = &d_long_state[(IndexType) thread << THREAD_SHIFT];
	uint32_t *ctx_a = d_ctx_a + thread * 4;
	uint32_t *ctx_b = d_ctx_b + thread * 4;
  uint32_t *a = shared_a + ((threadIdx.x >> 2) << 2);
  uint32_t *b = shared_b + ((threadIdx.x >> 2) << 2);
  uint32_t *c = shared_c + ((threadIdx.x >> 2) << 2);

  a[sub] = ctx_a[sub];
  b[sub] = ctx_b[sub];

  for ( i = start; i < end; ++i )
  {
    __syncthreads( );
    idx = (a[0] & 0x1FFFC0) >> 2;
    cl = loadGlobal32<uint32_t>(long_state + idx + sub);
    cl = ROTL32(cl, b[0] & 31);
    cl += loadGlobal32<uint32_t>(long_state + idx + sub + 4);
    cl = ROTL32(cl, b[1] & 31);
    cl += loadGlobal32<uint32_t>(long_state + idx + sub + 8);
    cl = ROTL32(cl, b[2] & 31);
    cl += loadGlobal32<uint32_t>(long_state + idx + sub + 12);
    cl = ROTL32(cl, b[3] & 31);
    c[sub] = cl;

    __syncthreads( );

    cl = a[sub]  ^ (t_fn0(c[sub] & 0xff) ^ t_fn1((c[(sub + 1) & 3] >> 8) & 0xff) ^ t_fn2((c[(sub + 2) & 3] >> 16) & 0xff) ^ t_fn3((c[(sub + 3) & 3] >> 24)));

    c[sub] = cl;

    bl = b[sub];

    long_state[idx + sub] ^= ROTL32(bl, a[0] & 31) + cl;
    long_state[idx + sub + 4] ^= ROTL32(bl, a[1] & 31) + cl;
    long_state[idx + sub + 8] ^= ROTL32(bl, a[2] & 31) + cl;
    long_state[idx + sub + 12] ^= ROTL32(bl, a[3] & 31) + cl;

    __syncthreads( );

    idx = (c[0] & 0x1FFFC0) >> 2;
    tmpl = loadGlobal32<uint32_t>(long_state + idx + sub);
    tmpl = ROTL32(tmpl, a[0] & 31);
    tmpl -= loadGlobal32<uint32_t>(long_state + idx + sub + 4);
    tmpl = ROTL32(tmpl, a[1] & 31);
    tmpl -= loadGlobal32<uint32_t>(long_state + idx + sub + 8);
    tmpl = ROTL32(tmpl, a[2] & 31);
    tmpl -= loadGlobal32<uint32_t>(long_state + idx + sub + 12);
    tmpl = ROTL32(tmpl, a[3] & 31);

    al = a[sub];
    al += cl * tmpl;
    a[sub] = tmpl;

    __syncthreads( );

    long_state[idx + sub] ^= ROTL32(al, a[0] & 31);
    long_state[idx + sub + 4] ^= ROTL32(al, a[1] & 31);
    long_state[idx + sub + 8] ^= ROTL32(al, a[2] & 31);
    long_state[idx + sub + 12] ^= ROTL32(al, a[3] & 31);

     a[sub] ^= al;
     b[sub] = cl;
  }

  if ( bfactor > 0 )
  {
    ctx_a[sub] = a[sub];
    ctx_b[sub] = b[sub];
  }

}

template<size_t ITERATIONS, uint32_t THREAD_SHIFT>
__global__ void cryptonight_core_gpu_phase3( int threads, int bfactor, int partidx, const uint32_t * __restrict__ long_state, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	int thread = ( blockDim.x * blockIdx.x + threadIdx.x ) >> 3;
	int sub = ( threadIdx.x & 7 ) << 2;

	const int batchsize = ITERATIONS >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40], text[4];
	MEMCPY8( key, d_ctx_key2 + thread * 40, 20 );
	MEMCPY8( text, d_ctx_state + thread * 50 + sub + 16, 2 );

	__syncthreads( );
	for ( int i = start; i < end; i += 32 )
	{
#pragma unroll
		for ( int j = 0; j < 4; ++j )
			text[j] ^= long_state[((IndexType) thread << THREAD_SHIFT) + (sub + i + j)];

		cn_aes_pseudo_round_mut( sharedMemory, text, key );
	}

	MEMCPY8( d_ctx_state + thread * 50 + sub + 16, text, 2 );
}

template<size_t ITERATIONS, uint32_t MASK, uint32_t THREAD_SHIFT>
void cryptonight_core_gpu_hash(nvid_ctx* ctx)
{
	dim3 grid( ctx->device_blocks );
	dim3 block( ctx->device_threads );
	dim3 block4( ctx->device_threads << 2 );
	dim3 block8( ctx->device_threads << 3 );

	int partcount = 1 << ctx->device_bfactor;

	/* bfactor for phase 1 and 3
	 *
	 * phase 1 and 3 consume less time than phase 2, therefore we begin with the
	 * kernel splitting if the user defined a `bfactor >= 5`
	 */
	int bfactorOneThree = ctx->device_bfactor - 4;
	if( bfactorOneThree < 0 )
		bfactorOneThree = 0;

	int partcountOneThree = 1 << bfactorOneThree;

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase1<ITERATIONS,THREAD_SHIFT><<< grid, block8 >>>( ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state, ctx->d_ctx_state, ctx->d_ctx_key1 ));

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}
	if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );

	for ( int i = 0; i < partcount; i++ )
	{
        CUDA_CHECK_MSG_KERNEL(
			ctx->device_id,
			"\n**suggestion: Try to increase the value of the attribute 'bfactor' or \nreduce 'threads' in the NVIDIA config file.**",
			cryptonight_core_gpu_phase2<ITERATIONS,THREAD_SHIFT,MASK><<<
				grid,
				block4,
				block4.x * sizeof(uint32_t) * 3
			>>>(
				ctx->device_blocks*ctx->device_threads,
				ctx->device_bfactor,
				i,
				ctx->d_long_state,
				ctx->d_ctx_a,
				ctx->d_ctx_b
			)
	    );

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		CUDA_CHECK_KERNEL(ctx->device_id, cryptonight_core_gpu_phase3<ITERATIONS,THREAD_SHIFT><<< grid, block8 >>>( ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state,
			ctx->d_ctx_state, ctx->d_ctx_key2 ));
	}
}

void cryptonight_core_cpu_hash(nvid_ctx* ctx, bool mineMonero)
{
#ifndef CONF_NO_MONERO
	if(mineMonero)
	{
		cryptonight_core_gpu_hash<MONERO_ITER, MONERO_MASK, 19u>(ctx);
	}
#endif
#ifndef CONF_NO_AEON
	if(!mineMonero)
	{
		cryptonight_core_gpu_hash<AEON_ITER, AEON_MASK, 18u>(ctx);
	}
#endif
}
