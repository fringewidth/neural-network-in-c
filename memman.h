#pragma once

typedef struct free_stack {
    void** freebuf;
    int frees_idx;
    int frees_max;
} freestack;

extern freestack frees;

void push_free(freestack*, void*);

void free_them_all();