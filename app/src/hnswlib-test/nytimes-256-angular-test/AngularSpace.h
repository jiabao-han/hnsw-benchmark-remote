//
// Created by root on 7/16/24.
//

#ifndef HNSW_BENCHMARK_ANGULARSPACE_H
#define HNSW_BENCHMARK_ANGULARSPACE_H

#pragma once
#include "hnswlib.h"
#include <cmath>

namespace hnswlib {

static float
AngularDistance(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    float norm1 = 0;
    float norm2 = 0;
    for (size_t i = 0; i < qty; i++) {
        res += *pVect1 * *pVect2;
        norm1 += *pVect1 * *pVect1;
        norm2 += *pVect2 * *pVect2;
        pVect1++;
        pVect2++;
    }
    return 1.0f - res / (sqrt(norm1) * sqrt(norm2));
}

#if defined(USE_AVX)
static float
AngularDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 sum = _mm256_set1_ps(0);
    __m256 norm1 = _mm256_set1_ps(0);
    __m256 norm2 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
        norm1 = _mm256_add_ps(norm1, _mm256_mul_ps(v1, v1));
        norm2 = _mm256_add_ps(norm2, _mm256_mul_ps(v2, v2));

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
        norm1 = _mm256_add_ps(norm1, _mm256_mul_ps(v1, v1));
        norm2 = _mm256_add_ps(norm2, _mm256_mul_ps(v2, v2));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    _mm256_store_ps(TmpRes, norm1);
    float n1 = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    _mm256_store_ps(TmpRes, norm2);
    float n2 = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return 1.0f - res / (sqrt(n1) * sqrt(n2));
}
#endif

class AngularSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    AngularSpace(size_t dim) {
        fstdistfunc_ = AngularDistance;
#if defined(USE_AVX)
        if (AVXCapable() && dim % 16 == 0)
            fstdistfunc_ = AngularDistanceSIMD16ExtAVX;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~AngularSpace() {}

    void normalize(float* vec) {
        float norm = 0;
        for (size_t i = 0; i < dim_; i++) {
            norm += vec[i] * vec[i];
        }
        norm = sqrt(norm);
        for (size_t i = 0; i < dim_; i++) {
            vec[i] /= norm;
        }
    }
};

}  // namespace hnswlib

#endif // HNSW_BENCHMARK_ANGULARSPACE_H
