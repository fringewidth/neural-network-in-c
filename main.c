#include "memman.h"
#include "layer.h"
#include "mtrx.h"
#include <stdio.h>
#include <stdlib.h>
#include "matfns.h"
#include <time.h>

freestack frees;

int main() {
    srand((unsigned)time(NULL));

    frees.frees_idx = 0;
    frees.frees_max = 128;
    frees.freebuf = malloc(frees.frees_max * sizeof(void*));
    if (!frees.freebuf) {
        free_them_all();
        return 1;
    }

    mat* mata = init_mat(4, 1);
    if (!mata) return 1;
    mat* matb = init_mat(4, 4);
    if (!matb) return 1;
    
    // Test width broadcasting: 4x1 + 4x4 should give 4x4 matrix of 2s
    mat* matc = mtrx_elemwise(mata, matb, add);
    if (!matc) return 1;
    printf("Width broadcasting test (4x1 + 4x4):\n");
    pprint(matc);
    
    // Test height broadcasting: 1x4 + 4x4 should also give 4x4 matrix of 2s
    mat* mata_h = init_mat(1, 4);  // 1x4 matrix
    if (!mata_h) return 1;
    mat* matc_h = mtrx_elemwise(mata_h, matb, add);
    if (!matc_h) return 1;
    printf("Height broadcasting test (1x4 + 4x4):\n");
    pprint(matc_h);

    free_them_all();
    return 0;
}
