// Henrique Noronha Facioli
// http://blog.cuvilib.com/2011/01/20/how-to-apply-filters-to-images-using-cuda/
// Using CUDA
// https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/shared-memory/shared-memory.cu
//
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5
#define CUDA_GRID 16

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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

__global__ void smoothing(PPMPixel *image, PPMPixel *image_copy, int img_x, int img_y) {
    int x, y;
    int p_x = threadIdx.x + blockDim.x * blockIdx.x;
    int p_y = threadIdx.y + blockDim.y * blockIdx.y;

    // Guarantee no memory access error
    if (p_x >= img_x || p_y >= img_y)
        return;

    __shared__ PPMPixel img_shrd[CUDA_GRID+MASK_WIDTH-1][CUDA_GRID+MASK_WIDTH-1];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Top-Left position
        for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
            for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
                int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
                int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
                int global_unique_value = (local_y * img_x) + local_x;

                if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
                    img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                    img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                    img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
                } else {
                    img_shrd[shared_x][shared_y].red = 0;
                    img_shrd[shared_x][shared_y].green = 0;
                    img_shrd[shared_x][shared_y].blue = 0;
                }
            } 
        }

    } else if (threadIdx.x == CUDA_GRID - 1 && threadIdx.y == 0) {
        // Top-Right position
        for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
            for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
                int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
                int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
                int global_unique_value = (local_y * img_x) + local_x;

                if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
                    img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                    img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                    img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
                } else {
                    img_shrd[shared_x][shared_y].red = 0;
                    img_shrd[shared_x][shared_y].green = 0;
                    img_shrd[shared_x][shared_y].blue = 0;
                }
            }
        }
    } else if (threadIdx.x == 0 && threadIdx.y == CUDA_GRID - 1) {
        // Botton-Left position
        for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
            for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
                int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
                int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
                int global_unique_value = (local_y * img_x) + local_x;

                if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
                    img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                    img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                    img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
                } else {
                    img_shrd[shared_x][shared_y].red = 0;
                    img_shrd[shared_x][shared_y].green = 0;
                    img_shrd[shared_x][shared_y].blue = 0;
                }
            }
        }
    
    } else if (threadIdx.x == CUDA_GRID - 1 && threadIdx.y == CUDA_GRID - 1) {
        // Botton-Right position
        for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
            for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
                int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
                int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
                int global_unique_value = (local_y * img_x) + local_x;

                if (local_x >= 0 && local_x < img_x && local_y >= 0 && local_y < img_y) {
                    img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                    img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                    img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
                } else {
                    img_shrd[shared_x][shared_y].red = 0;
                    img_shrd[shared_x][shared_y].green = 0;
                    img_shrd[shared_x][shared_y].blue = 0;
                }
            }
        }

    } else if (threadIdx.x == 0) {
        // First Column
        for (int local_x = p_x - (MASK_WIDTH-1)/2; local_x <= p_x; local_x++) {
            int shared_x = local_x + (MASK_WIDTH-1)/2 - p_x;
            int shared_y = (MASK_WIDTH-1)/2  + threadIdx.y;
            int global_unique_value = (p_y * img_x) + local_x;
            // printf("Estive 1 %d %d\n", img_x, global_unique_value);
            if (local_x >= 0 && local_x < img_x) {
                img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
            } else {
                img_shrd[shared_x][shared_y].red = 0;
                img_shrd[shared_x][shared_y].green = 0;
                img_shrd[shared_x][shared_y].blue = 0;
            }
        }
    } else if (threadIdx.x == CUDA_GRID - 1) {
        // Last Column
        for (int local_x = p_x; local_x <= p_x + (MASK_WIDTH - 1)/2; local_x++) {
            int shared_x = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_x - p_x);
            int shared_y = (MASK_WIDTH-1)/2  + threadIdx.y;
            int global_unique_value = (p_y * img_x) + local_x;
            if (local_x >= 0 && local_x < img_x) {
                img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
            } else {
                img_shrd[shared_x][shared_y].red = 0;
                img_shrd[shared_x][shared_y].green = 0;
                img_shrd[shared_x][shared_y].blue = 0;
            }
        }
    } else if (threadIdx.y == 0) {
        // First Line
        for (int local_y = p_y-(MASK_WIDTH-1)/2; local_y <= p_y; local_y++) {
            int shared_x = (MASK_WIDTH-1)/2 + threadIdx.x;
            int shared_y = local_y + (MASK_WIDTH-1)/2 - p_y;
            int global_unique_value = (local_y * img_x) + p_x;
            if (local_y >= 0 && local_y < img_y) {
                img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
            } else {
                img_shrd[shared_x][shared_y].red = 0;
                img_shrd[shared_x][shared_y].green = 0;
                img_shrd[shared_x][shared_y].blue = 0;
            }
        }
    } else if (threadIdx.y == CUDA_GRID - 1) {
        // Last Line
        for (int local_y = p_y; local_y <= p_y + (MASK_WIDTH - 1)/2; local_y++) {
            int shared_x = (MASK_WIDTH-1)/2 + threadIdx.x;
            int shared_y = (MASK_WIDTH-1)/2 + (CUDA_GRID - 1) + (local_y - p_y);
            int global_unique_value = (local_y * img_x) + p_x;
            if (local_y >= 0 && local_y < img_y) {
                img_shrd[shared_x][shared_y].red = image_copy[global_unique_value].red;
                img_shrd[shared_x][shared_y].green = image_copy[global_unique_value].green;
                img_shrd[shared_x][shared_y].blue = image_copy[global_unique_value].blue;
            } else {
                img_shrd[shared_x][shared_y].red = 0;
                img_shrd[shared_x][shared_y].green = 0;
                img_shrd[shared_x][shared_y].blue = 0;
            }
        }
    } else {
        // All elements except the borders
        int shared_x = threadIdx.x+(MASK_WIDTH-1)/2;
        int shread_y = threadIdx.y+ (MASK_WIDTH-1)/2;
        img_shrd[shared_x][shread_y].red = image_copy[(p_y * img_x) + p_x].red;
        img_shrd[shared_x][shread_y].green = image_copy[(p_y * img_x) + p_x].green;
        img_shrd[shared_x][shread_y].blue = image_copy[(p_y * img_x) + p_x].blue;
    }
    
    __syncthreads();

    int total_red = 0 , total_blue = 0, total_green = 0;
    for (y = threadIdx.y; y <= threadIdx.y + MASK_WIDTH-1; y++) {
        for (x = threadIdx.x; x <= threadIdx.x + MASK_WIDTH-1; x++) {
            if (x >= 0 && y >= 0 && y < CUDA_GRID+MASK_WIDTH && x < CUDA_GRID+MASK_WIDTH) {
                total_red += img_shrd[x][y].red;
                total_blue += img_shrd[x][y].blue;
                total_green += img_shrd[x][y].green;
            } //if
        } //for z
    } //for y
    image[(p_y * img_x) + p_x].red = total_red / (MASK_WIDTH*MASK_WIDTH);
    image[(p_y * img_x) + p_x].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
    image[(p_y * img_x) + p_x].green = total_green / (MASK_WIDTH*MASK_WIDTH);
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    double t_start, t_end;
    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    // Starting allocating memory on GPU
    int n = (image->x * image->y);
    PPMPixel *d_image ,*d_image_output, *img_out;

    // Malloc
    
    img_out = (PPMPixel *)calloc(n, sizeof(PPMPixel));
    t_start = rtclock();
    cudaMalloc((void **) &d_image, sizeof(PPMPixel) * n);
    cudaMalloc((void **) &d_image_output, sizeof(PPMPixel) * n);

    // Copy
    cudaMemcpy(d_image, image->data, sizeof(PPMPixel)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_output, image_output->data, sizeof(PPMPixel)*n, cudaMemcpyHostToDevice);    

    int nrow = ceil((double) image->x / CUDA_GRID);    // Create the total number of blocs necessary
    int nlin = ceil((double) image->y / CUDA_GRID);    // to iterate through matrix of size 32
    dim3 gridDim(nrow, nlin);                          // Now, iterate on a grid at most 32x32
    dim3 blockDim(CUDA_GRID, CUDA_GRID);               //  32 x 32 x 1 (1024)

    smoothing<<<gridDim, blockDim>>>(d_image_output, d_image, image->x, image->y);
    cudaDeviceSynchronize();

    cudaMemcpy(img_out, d_image_output, sizeof(PPMPixel) * n, cudaMemcpyDeviceToHost); 
    t_end = rtclock();
    image_output->data = img_out;
    fprintf(stdout, "%0.6lf\n", t_end - t_start);  

    writePPM(image_output);

    free(image);
    free(image_output);
}
