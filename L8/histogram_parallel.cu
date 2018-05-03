// Henrique Noronha Facioli
// 157986

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

__global__ void histogram_sum(PPMPixel *linear_image, float image_size, float *h) {
	int j, k, l;

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= image_size)
		return;
	
	for (j = 0; j <= 3; j++) {
		for (k = 0; k <= 3; k++) {
			for (l = 0; l <= 3; l++) {
				if (linear_image[i].red   == j &&
					linear_image[i].green == k &&
					linear_image[i].blue  == l
				) {
					atomicAdd(&(h[(l+4*k+16*j)]), 1.0);
				}
			}
		}
	}
}

void Histogram(PPMImage *image, float *h) {
	int i;
	float n = image->y * image->x;
	PPMPixel *d_image;
	float *d_h;
	double t_start, t_end;

	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

	t_start = rtclock();
	cudaMalloc((void **) &d_image, sizeof(PPMPixel) * n);
	cudaMalloc((void **) &d_h, sizeof(float) * 64);
	t_end = rtclock();
	fprintf(stdout, "criar_buff: %0.6lf\n", t_end - t_start);
	
	t_start = rtclock();
	cudaMemset(d_h, 0.0, sizeof(float) * 64);
	cudaMemcpy(d_image, image->data, sizeof(PPMPixel)*n, cudaMemcpyHostToDevice);
	t_end = rtclock();
	fprintf(stdout, "enviar: %0.6lf\n", t_end - t_start);
	
	int n_blocks = ceil(n / 1024.0);
	dim3 gridDim(n_blocks);
	dim3 blockDim(1024);


	histogram_sum<<<gridDim, blockDim>>>(d_image, n, d_h);
	t_end = rtclock();
	fprintf(stdout, "kernel: %0.6lf\n", t_end - t_start);

	t_start = rtclock();
	cudaMemcpy(h, d_h, sizeof(float) * 64, cudaMemcpyDeviceToHost);
	t_end = rtclock();
	fprintf(stdout, "receber: %0.6lf\n", t_end - t_start);
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) h[i] = 0.0;

	t_start = rtclock();
	Histogram(image, h);
	t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]/(image->x * image->y));
	}
	printf("\n");
	fprintf(stdout, "%0.6lf\n", t_end - t_start);  
	free(h);
}
