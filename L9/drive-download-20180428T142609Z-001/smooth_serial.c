#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5

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


void Smoothing_CPU_Serial(PPMImage *image, PPMImage *image_copy) {
    int i, j, y, x;
    int total_red, total_blue, total_green;

    for (i = 0; i < image->y; i++) {
        for (j = 0; j < image->x; j++) {
            total_red = total_blue = total_green = 0;
            for (y = i - ((MASK_WIDTH-1)/2); y <= (i + ((MASK_WIDTH-1)/2)); y++) {
                for (x = j - ((MASK_WIDTH-1)/2); x <= (j + ((MASK_WIDTH-1)/2)); x++) {
                    if (x >= 0 && y >= 0 && y < image->y && x < image->x) {
                        total_red += image_copy->data[(y * image->x) + x].red;
                        total_blue += image_copy->data[(y * image->x) + x].blue;
                        total_green += image_copy->data[(y * image->x) + x].green;
                    } //if
                } //for z
            } //for y
            image->data[(i * image->x) + j].red = total_red / (MASK_WIDTH*MASK_WIDTH);
            image->data[(i * image->x) + j].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
            image->data[(i * image->x) + j].green = total_green / (MASK_WIDTH*MASK_WIDTH);
        }
    }
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    double t_start, t_end;
    int i;
    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    t_start = rtclock();
    Smoothing_CPU_Serial(image_output, image);
    t_end = rtclock();

    writePPM(image_output);

    //fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
    free(image);
    free(image_output);
}
