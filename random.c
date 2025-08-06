#include <math.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

static float runiform(void) {
    return ((float)rand() + 0.5) / ((float)RAND_MAX + 1.0);
}

float rnormal(const float mean, const float std) {
    float u1 = runiform();
    float u2 = runiform();
    float z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return z0 * std + mean;
}
