#pragma once
typedef struct matrix {
    int h;
    int w;
    float* data;
    float* (*idxfn)(const struct matrix* mtrx, const int i, const int j);
} mat;

float* idx_mat(const mat*, const int, const int);

float* idx_mat_t(const mat*, const int, const int);

mat* init_mat(const int, const int);

mat* init_mat_random(const int, const int, float, float);

mat* tr(const mat*);

mat* mtrx_elemwise(mat*, mat*, float (*)(float*, float*));

void mtrx_elemwise_ip(mat* a, mat* b, void (*)(float*, float*));

void mtrx_elemwise_ip_unary(mat* a, void (*)(float*));

mat* mm(const mat* a, const mat* b);

void pprint(const mat*);
