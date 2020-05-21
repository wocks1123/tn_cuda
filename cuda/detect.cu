#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _RESULT
{
    float score;
    int match;
} RESULT;

typedef struct _DIM
{
    int x;
    int y;
    int z;
} DIM;

typedef struct _INDEX
{
    int i;
    int j;
} INDEX;

#define ARR_I_J(arr, i, j) arr[i * K + j]

__global__ void detect_kernel(const int* ref_index, const float* ref_score, INDEX* index, float* score, const size_t L, const size_t K,
                            const float score_thr, const int tmp_wnd, const int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int len = tid + offset;

    if (tid >= K)
    {
        return;
    }

    for (int i = 1; i < L; i++)
    {
        for (int j = tid; j < len; j++)
        {
            INDEX t;
            t.i = -1;
            t.j = -1;
            ARR_I_J(index, i, j) = t;

            float curr_score = ARR_I_J(ref_score, i, j);
            int curr_index = ARR_I_J(ref_index, i, j);

            if (curr_score < score_thr)
                continue;

            float max_score = curr_score;
            int prev_i = -1;
            int prev_j = -1;

            int start_idx = i - tmp_wnd >= 0 ? i - tmp_wnd : 0;
            for (int l = start_idx; l < i; l++)
            {
                for (int k = 0; k < K; k++)
                {
                    float prev_score = ARR_I_J(score, l, k);
                    int prev_index = ARR_I_J(ref_index, l, k);

                    if (prev_index > curr_index)
                        continue;

                    if (prev_index < curr_index - tmp_wnd)
                        continue;

                    if (max_score < prev_score + curr_score)
                    {
                        max_score = prev_score + curr_score;
                        prev_i = l;
                        prev_j = k;
                    }
                }
            }

            ARR_I_J(score, i, j) = max_score;
            INDEX r;
            r.i = prev_i;
            r.j = prev_j;
            ARR_I_J(index, i, j) = r;
        }
        __syncthreads();
    }
}

extern "C" {

void reverse_arr(int* arr, size_t length)
{
    for(int i = 0; i < length / 2; i++)
    {
        int temp = arr[length - 1 - i];
        arr[length - 1 - i] = arr[i];
        arr[i] = temp;
    }
}

void call_kernel(const int* _ref_index, const float* _ref_score, size_t L, size_t K, float score_thr, int tmp_wnd,
            int* query_index, int* path, int n_block, int n_thread, RESULT* res)
{
    int* ref_index;
    float* ref_score;
    RESULT* ret;
    cudaMalloc((void**)&ref_index, L * K * sizeof(int));
    cudaMalloc((void**)&ref_score, L * K * sizeof(float));
    cudaMalloc((void**)&ret, sizeof(RESULT));
    cudaMemcpy(ref_index, _ref_index, L * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ref_score, _ref_score, L * K * sizeof(float), cudaMemcpyHostToDevice);

    INDEX* d_index, *h_index;
    float* d_score, *h_score;

    h_index = (INDEX*)calloc(sizeof(INDEX), L * K);
    h_score = (float*)calloc(sizeof(float), L * K);

    for (int i = 0; i < K; i++)
    {
        h_score[i] = ARR_I_J(_ref_score, 0, i);
        INDEX t;
        t.i = -1;
        t.j = -1;
        h_index[i] = t;
    }

    cudaMalloc((void**)&d_index, L * K * sizeof(INDEX));
    cudaMalloc((void**)&d_score, L * K * sizeof(float));
    cudaMemcpy(d_index, h_index, L * K * sizeof(INDEX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_score, h_score, L * K * sizeof(float), cudaMemcpyHostToDevice);

    int offset = K / (n_block * n_thread);
    offset = offset ? offset : 1;

    detect_kernel <<< n_block, n_thread >>> (ref_index, ref_score, d_index, d_score, L, K, score_thr, tmp_wnd, offset);

    cudaMemcpy(h_index, d_index, L * K * sizeof(INDEX), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_score, d_score, L * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res, ret, sizeof(RESULT), cudaMemcpyDeviceToHost);


    float max_score = -1.0f;
    INDEX max_index;
    max_index.i = -1;
    max_index.j = -1;
    for(int i = L - 1; i >= 0; i--)
    {
        for(int j = 0; j < K; j++)
        {
            if (max_score < ARR_I_J(h_score, i, j))
            {
                max_score = ARR_I_J(h_score, i, j);
                max_index.i = i;
                max_index.j = j;
            }
        }
    }

    int idx = 0;
    int p_i = max_index.i;
    int p_j = max_index.j;

    while(p_i != -1)
    {
        INDEX t = ARR_I_J(h_index, p_i, p_j);
        query_index[idx] = p_i;
        path[idx++] = ARR_I_J(_ref_index, p_i, p_j);
        p_i = t.i;
        p_j = t.j;
    }

    res->score = max_score;
    res->match = idx;

    reverse_arr(query_index, res->match);
    reverse_arr(path, res->match);

    free(h_score);
    free(h_index);

    cudaFree(d_score);
    cudaFree(d_index);
    cudaFree(ref_index);
    cudaFree(ref_score);
    cudaFree(ret);
}

}