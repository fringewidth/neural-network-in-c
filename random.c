#include <math.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

static double runiform(void) {
    return ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0);
}

double rnormal(double mean, double stddev) {
    double u1 = runiform();
    double u2 = runiform();
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return z0 * stddev + mean;
}