#pragma once

static inline float add(float* a, float* b) { return *a + *b; }

static inline void add_ip(float* a, float* b) { *a += *b; }

static inline float sub(float* a, float* b) { return *a - *b; }

static inline void sub_ip(float* a, float* b) { *a -= *b; }

static inline float mul(float* a, float* b) { return *a * *b; }

static inline void mul_ip(float* a, float* b) { *a *= *b; }
