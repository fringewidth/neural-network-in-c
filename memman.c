#include "memman.h"
#include <stdlib.h>

void push_free(freestack* frees, void* ptr) {
    if (frees->frees_idx >= frees->frees_max) {
        frees->frees_max *= 2;
        void* new_buf = realloc(frees->freebuf, frees->frees_max * sizeof(void*));
        if (!new_buf) {
            return;
        }
        frees->freebuf = new_buf;
    }
    frees->freebuf[frees->frees_idx++] = ptr;
}

void free_them_all() {
    int len = frees.frees_idx;
    while (len--) {
        free(frees.freebuf[len]);
    }
    free(frees.freebuf);
}
