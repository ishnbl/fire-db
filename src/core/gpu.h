#pragma once
#ifndef FIREDB_GPU_H
#define FIREDB_GPU_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "slab.h"


struct SearchResult {
    uint64_t id;
    float score;
};


__global__ void compute_norms_kernel(const float* data, float* norms, int rows, int cols) {
    // BlockIdx is provided by CUDA, we do not need to pass it
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    float sum = 0.0f;
    const float* vector = data + (idx * cols);

    for (int i = 0; i < cols; i++) {
        float val = vector[i];
        sum += val * val;
    }
    norms[idx] = sum;
}


__global__ void compute_l2_dist_kernel(const float* db_norms, const float* query_norms, float* d_dot_products, int num_db, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_db * num_queries;

    if (idx >= total) return;


    int row = idx % num_db;
    int col = idx / num_db;

    float val = d_dot_products[idx];

    d_dot_products[idx] = val + db_norms[row] + query_norms[col];
}



class GpuIndex {
private:
    cublasHandle_t handle;

    float* d_db = nullptr;
    float* d_db_norms = nullptr;

    float* d_queries = nullptr;
    float* d_q_norms = nullptr;
    float* d_results = nullptr;

    size_t max_vectors;
    size_t dim;
    size_t current_count = 0;
    size_t max_batch_size = 100;

public:
    GpuIndex(size_t dimension, size_t capacity) : dim(dimension), max_vectors(capacity) {
        cublasCreate(&handle);


        cudaMalloc(&d_db, max_vectors * dim * sizeof(float));
        cudaMalloc(&d_db_norms, max_vectors * sizeof(float));
        cudaMalloc(&d_queries, max_batch_size * dim * sizeof(float));
        cudaMalloc(&d_q_norms, max_batch_size * sizeof(float));
        cudaMalloc(&d_results, max_vectors * max_batch_size * sizeof(float));
    }

    ~GpuIndex() {
        cudaFree(d_db);
        cudaFree(d_db_norms);
        cudaFree(d_queries);
        cudaFree(d_q_norms);
        cudaFree(d_results);
        cublasDestroy(handle);
    }

    bool add_single_vector(const float* host_vec) {
        if (current_count >= max_vectors) {
            std::cout << "GPU Full!" << std::endl;
            return false;
        }

        size_t offset = current_count * dim;
        cudaMemcpy(d_db + offset, host_vec, dim * sizeof(float), cudaMemcpyHostToDevice);

        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) sum_sq += host_vec[i] * host_vec[i];

        cudaMemcpy(d_db_norms + current_count, &sum_sq, sizeof(float), cudaMemcpyHostToDevice);

        current_count++;
        return true;
    }

    void load_data(const MatrixSlab& slab) {
        current_count = slab.get_count();
        std::cout << "[GPU] Uploading " << current_count << " vectors..." << std::endl;

        cudaMemcpy(d_db, slab.get_data_ptr(),
                   current_count * dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        compute_norms_kernel<<<blocks, threads>>>(d_db, d_db_norms, current_count, dim);

        cudaDeviceSynchronize();
    }

    std::vector<std::vector<SearchResult>> search(const std::vector<std::vector<float>>& queries, int k) {
        int num_queries = queries.size();
        if (num_queries == 0) return {};


        std::vector<float> flat_queries;
        std::vector<float> host_q_norms;

        for (const auto& q : queries) {
            float sum_sq = 0.0f;
            for (float val : q) {
                flat_queries.push_back(val);
                sum_sq += val * val;
            }
            host_q_norms.push_back(sum_sq);
        }

        cudaMemcpy(d_queries, flat_queries.data(), flat_queries.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q_norms, host_q_norms.data(), host_q_norms.size() * sizeof(float), cudaMemcpyHostToDevice);

        float alpha = -2.0f;
        float beta = 0.0f;

        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            current_count, num_queries, dim,
            &alpha,
            d_db, dim,
            d_queries, dim,
            &beta,
            d_results, current_count
        );


        int total_pairs = current_count * num_queries;
        int threads = 256;
        int blocks = (total_pairs + threads - 1) / threads;

        compute_l2_dist_kernel<<<blocks, threads>>>(
            d_db_norms, d_q_norms, d_results, current_count, num_queries
        );
        cudaDeviceSynchronize();


        std::vector<float> all_scores(total_pairs);
        cudaMemcpy(all_scores.data(), d_results, total_pairs * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<std::vector<SearchResult>> final_results(num_queries);

        for (int q = 0; q < num_queries; q++) {
            float* query_scores = all_scores.data() + (q * current_count);

            std::vector<std::pair<float, int>> candidates(current_count);
            for (int i = 0; i < current_count; i++) {
                candidates[i] = { query_scores[i], i };
            }

            int safe_k = std::min((size_t)k, current_count);
            std::partial_sort(candidates.begin(), candidates.begin() + safe_k, candidates.end());

            for (int i = 0; i < safe_k; i++) {
                final_results[q].push_back({ (uint64_t)candidates[i].second, candidates[i].first });
            }
        }

        return final_results;
    }

    std::vector<SearchResult> search_one(const std::vector<float>& query, int k) {
        return search({query}, k)[0];
    }
};

#endif