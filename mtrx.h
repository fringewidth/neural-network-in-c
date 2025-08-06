#pragma once

struct matrix;

typedef float* (*idx_fn_ptr)(const struct matrix* mtrx, const int i, const int j);


typedef struct matrix {
    int h;
    int w;
    float* data;
    idx_fn_ptr idxfn;
} mat;

mat* init_mat(const int h, const int w);

mat* init_mat_random(const int h, const int w, float mean, float std);

mat* tr(const mat* mtrx);

mat* mtrx_elemwise(mat* a, mat* b, float (*op)(const float*, const float*));

void mtrx_elemwise_ip(mat* a, mat* b, void (*op)(float*, const float*));

void mtrx_elemwise_ip_unary(mat* a, void (*op)(float*));

mat* mm(const mat* a, const mat* b);

void pprint(const mat* mtrx);
