#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <windows.h>
#include "cpu_bitmap.h"

using namespace std;

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

// light position
float LX = 0.0f;
float LY = 1000.0f;
float LZ = 0.0f;

// camera position
float CameraX = 0.0f;
float CameraY = 0.0f;
float CameraZ = 10000.0f;

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

struct Sphere {
    float   ra,ba,ga;
	float   rd,bd,gd;
    float   radius;
    float   x,y,z;
	float hit(float ox, float oy, float* nx, float* ny, float* nz, float* lx, float* ly, float* lz, float* rx, float* ry, float* rz, float* vx, float* vy, float* vz){
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

Sphere s[SPHERES];

float trim(float n){
	if(n<0)	return 0;
	if(n>1) return 1;
	return n;
}

float max0(float n){
	if(n<0)	return 0;
	return n;
}

void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
	for(int x=0; x!=DIM; ++x){
		for(int y=0; y!=DIM; ++y){

			int offset = x + y * DIM;
			float   ox = (x - DIM/2);
			float   oy = (y - DIM/2);

			float   r=0, g=0, b=0;
			float   maxz = -INF;
			for(int i=0; i<SPHERES; i++) {
				float   nx, ny, nz, lx, ly, lz, rx, ry, rz, vx, vy, vz;
				float   t = s[i].hit( ox, oy, &nx, &ny, &nz, &lx, &ly, &lz, &rx, &ry, &rz, &vx, &vy, &vz);
				if (t > maxz) {

					float ln = max0(lx*nx+ly*ny+lz*nz);
					float rv = max0(pow(rx*vx+ry*vy+rz*vz, ALPHA));

					r = KAR*s[i].ra + KDR*ln*s[i].rd + KSR*rv*LSR;
					g = KAG*s[i].ga + KDG*ln*s[i].gd + KSG*rv*LSG;
					b = KAB*s[i].ba + KDB*ln*s[i].bd + KSB*rv*LSB;
					maxz = t;
				}
			} 
			ptr[offset*4 + 0] = (int)(trim(r) * 255);
			ptr[offset*4 + 1] = (int)(trim(g) * 255);
			ptr[offset*4 + 2] = (int)(trim(b) * 255);
			ptr[offset*4 + 3] = 255;
		}
	}
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    // capture the start time
	clock_t begin = clock();

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;

    // allocate memory on the GPU for the output bitmap
	dev_bitmap = (unsigned char*)malloc(bitmap.image_size());

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
	memcpy(s, temp_s, sizeof(Sphere) * SPHERES);
    free( temp_s );

    // generate a bitmap from our sphere data
	kernel(bitmap.pixels);

	clock_t end = clock();
	float elapsedTime = float(end - begin)*1000 / CLOCKS_PER_SEC;

	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	free(dev_bitmap);

    // display
    bitmap.display_and_exit();
}