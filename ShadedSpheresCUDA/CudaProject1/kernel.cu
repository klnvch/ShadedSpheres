/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cpu_bitmap.h"

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere {
    float   ra,ba,ga;
	float   rd,bd,gd;
    float   radius;
    float   x,y,z;
    __device__ float hit(float ox, float oy, float* nx, float* ny, float* nz, float* lx, float* ly, float* lz, float* rx, float* ry, float* rz, float* vx, float* vy, float* vz) {
        
		// light position
		float LX = 0.0f;
		float LY = 1000.0f;
		float LZ = 0.0f;

		// camera position
		float CameraX = 0.0f;
		float CameraY = 0.0f;
		float CameraZ = 10000.0f;
		
		float dx = ox - x;
        float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {

			// calculate normal vector
			float dz = sqrtf( radius*radius - dx*dx - dy*dy );
			float oz = z + dz;
			*nx = dx;
			*ny = dy;
			*nz = dz;
			float n1 = sqrtf((*nx)*(*nx)+(*ny)*(*ny)+(*nz)*(*nz));
			*nx /= n1;
			*ny /= n1;
			*nz /= n1;
			
			// calculate light vector
			*lx = LX - ox;
			*ly = LY - oy;
			*lz = LZ - oz;
			float n2 = sqrtf((*lx)*(*lx)+(*ly)*(*ly)+(*lz)*(*lz));
			*lx /= n2;
			*ly /= n2;
			*lz /= n2;

			// calculate reflection vector
			float n3 = 2.0f*((*nx)*(*lx)+(*ny)*(*ly)+(*nz)*(*lz));
			*rx = n3*(*nx) - (*lx);
			*ry = n3*(*ny) - (*ly);
			*rz = n3*(*nz) - (*lz);
			float n4 = sqrtf((*rx)*(*rx)+(*ry)*(*ry)+(*rz)*(*rz));
			*rx /= n4;
			*ry /= n4;
			*rz /= n4;

			// calculate camera vector
			*vx = CameraX - ox;
			*vy = CameraY - oy;
			*vz = CameraZ - oz;
			float n5 = sqrtf((*vx)*(*vx)+(*vy)*(*vy)+(*vz)*(*vz));
			*vx /= n5;
			*vy /= n5;
			*vz /= n5;
			return dz + z;
		}
		return -INF;
    }
};
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel( unsigned char *ptr ) {

	// colors
	float LSR = 0.55f; 
	float LSG = 0.55f; 
	float LSB = 0.55f;

	// strenght
	float KAR = 0.3f;
	float KAG = 0.3f; 
	float KAB = 0.3f; 
	
	float KDR = 1.0f; 
	float KDG = 1.0f; 
	float KDB = 1.0f; 

	float KSR = 0.8f; 
	float KSG = 0.8f; 
	float KSB = 0.8f;
	//alfa
	float ALPHA = 15.0f;

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

	float   r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<SPHERES; i++) {
		float   nx, ny, nz, lx, ly, lz, rx, ry, rz, vx, vy, vz;
		float   t = s[i].hit( ox, oy, &nx, &ny, &nz, &lx, &ly, &lz, &rx, &ry, &rz, &vx, &vy, &vz);
		if (t > maxz) {

			float ln = max(0.0f, (lx*nx+ly*ny+lz*nz));
			float rv = max(0.0f, pow(rx*vx+ry*vy+rz*vz, ALPHA));

			r = KAR*s[i].ra + 
				KDR*ln*s[i].rd + KSR*rv*LSR;
			g = KAG*s[i].ga + 
				KDG*ln*s[i].gd + KSG*rv*LSG;
			b = KAB*s[i].ba + 
				KDB*ln*s[i].bd + KSB*rv*LSB;
			maxz = t;
		}
	} 
	ptr[offset*4 + 0] = (int)(min(1.0f, r) * 255);
	ptr[offset*4 + 1] = (int)(min(1.0f, g) * 255);
	ptr[offset*4 + 2] = (int)(min(1.0f, b) * 255);
	ptr[offset*4 + 3] = 255;
	
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].ra = temp_s[i].rd = rnd( 1.0f );
        temp_s[i].ga = temp_s[i].gd = rnd( 1.0f );
        temp_s[i].ba = temp_s[i].bd = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, sizeof(Sphere) * SPHERES) );
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( dev_bitmap );

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );

    // display
    bitmap.display_and_exit();
}