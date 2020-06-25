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

//#define N_DEBUG

struct PathNode
{
    int prevTimestamp;
    int prevQueryIdx;
    float score;

public:
    PathNode() : prevTimestamp(-1), prevQueryIdx(-1), score(0)
    {
    }

    PathNode(float _score) : prevTimestamp(-1), prevQueryIdx(-1), score(_score)
    {
    }
};

struct Lock{
  int *mutex;
  Lock(void){
    int state = 0;
    cudaMalloc((void**) &mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
  }
  ~Lock(void){
    cudaFree(mutex);
  }
  __device__ void lock(uint compare){
    while(atomicCAS(mutex, compare, 0xFFFFFFFF) != compare);    //0xFFFFFFFF is just a very large number. The point is no block index can be this big (currently).
  }
  __device__ void unlock(uint val){
    atomicExch(mutex, val+1);
  }
};

#define ARR_I_J(arr, i, j) arr[i * K + j]


__global__ void detect_kernel(const int* ref_index, const float* ref_score, INDEX* index, float* score, const size_t L, const size_t K,
                            const float score_thr, const int tmp_wnd, const int offset)
{
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //int len = tid + offset;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = tid * offset;
    int e_idx = s_idx + offset;
    if (e_idx > K)
        e_idx = K;

    if (tid >= K)
    {
        return;
    }

#ifdef N_DEBUG
    //printf("[tid:%d] %d ~ %d(%d)\n", tid, s_idx, e_idx, offset);
    //printf("[tid:%d] %d ~ %d(%d)\n", tid, s_idx, e_idx, offset);
#endif

    for (int i = 1; i < L; i++)
    {
        for (int j = s_idx; j < e_idx; j++)
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


__global__ void detect_cuda2(const int* d_ref_index, const float* d_ref_score, PathNode* d_PN, const int L, const int K, int tmp_wnd, const int offset,
                           int* d_last_query_idx, int* d_last_ref_idx, float* d_max_score
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = tid * offset;
    int e_idx = s_idx + offset;
    if (e_idx > K)
        e_idx = K;

#ifdef N_DEBUG
    printf("[tid:%d] %d ~ %d(%d)\n", tid, s_idx, e_idx, offset);
#endif

    if (tid >= K)
    {
        return;
    }

    for (int i = 1; i < L; i++)
    {
        for (int j = s_idx; j < e_idx; j++)
        {
            float curr_score = ARR_I_J(d_ref_score, i, j);
            int curr_timpstamp = ARR_I_J(d_ref_index, i, j);

            float max_score = curr_score;
            int prev_i = -1;    //query idx
            int prev_j = -1;    //ref timestamp

            int start_idx = i - tmp_wnd >= 0 ? i - tmp_wnd : 0;
#ifdef N_DEBUG
            //printf("start %d -> %d\n", start_idx, i);
#endif
            for (int l = start_idx; l < i; l++)
            {
                for (int k = 0; k < K; k++)
                {
                    float prev_score = ARR_I_J(d_PN, l, k).score; //prev까지 가는 path의 score
                    int prev_timestamp = ARR_I_J(d_ref_index, l, k);

                    if (prev_timestamp >= curr_timpstamp)
                        continue;

                    if (prev_timestamp <= curr_timpstamp - tmp_wnd)
                        continue;

                    if (max_score <= prev_score + curr_score)
                    {
                        //printf("updeted(prev score %f -> %f\n", max_score, prev_score + curr_score);
                        max_score = prev_score + curr_score;
                        prev_i = l;
                        prev_j = k;
                    }
                }
            }

            ARR_I_J(d_PN, i, j).prevQueryIdx = prev_i;
            ARR_I_J(d_PN, i, j).prevTimestamp = prev_j;
            ARR_I_J(d_PN, i, j).score = max_score;

            if (*d_max_score <= max_score)
            {
                *d_max_score = max_score;
                //d_last_queryidx_list[tid] = i;
                //h_last_refidx_list[tid] = j;
                *d_last_query_idx = i;
                *d_last_ref_idx = j;
            }
        }
        __syncthreads();
    }
    __syncthreads();
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

void reverse_arr_f(float* arr, size_t length)
{
    for(int i = 0; i < length / 2; i++)
    {
        float temp = arr[length - 1 - i];
        arr[length - 1 - i] = arr[i];
        arr[i] = temp;
    }
}

void call_kernel(const int* _ref_index, const float* _ref_score, size_t L, size_t K, float score_thr, int tmp_wnd,
            int* query_index, int* path, int n_block, int n_thread, RESULT* res)
{

#ifdef N_DEBUG
    printf("K : %lu\n", K);
#endif


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
    if (offset && K % (n_block * n_thread) != 0)
        offset += 1;
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


void call_kernel2(int* _ref_index, float* _ref_score, int L, int K, float score_thr, int tmp_wnd,
                int n_block, int n_thread, int* res_q, int* res_p, float* result_score_path, RESULT* res)
{
    int total_thread_cnt = n_block * n_thread;

    int* d_ref_index;
    float* d_ref_score;
    PathNode* d_PN;

    // (1, num_of_thread) array thread별 계산 결과 저장

    /*
    *   set variables
    */
    PathNode* h_PN = new PathNode[L * K];
    //printf("update first row\n");
    for (int i = 0; i < K; i++)
    {
        ARR_I_J(h_PN, 0, i).score = ARR_I_J(_ref_score, 0, i);
        //printf("%f ", ARR_I_J(_ref_score, 0, i));
    }
    //printf("\n");

#ifdef N_DEBUG
    printf("K : %d\n", K);
    printf("====== ARR_I_J(_ref_score, i, j)\n");
    for (int i = 0 ; i < L; i++)
    {
        printf("[%d]\t", i);
        for (int j = 0 ; j < K; j++)
        {
            printf("%f ", ARR_I_J(_ref_score, i, j));
        }
        printf("\n");
    }

#endif

    cudaMalloc((void**)&d_ref_index, L * K * sizeof(int));
    cudaMalloc((void**)&d_ref_score, L * K * sizeof(float));
    cudaMalloc((void**)&d_PN, L * K * sizeof(PathNode));

    cudaMemcpy(d_ref_index, _ref_index, L * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_score, _ref_score, L * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PN, h_PN, L * K * sizeof(PathNode), cudaMemcpyHostToDevice);

    int* d_last_query_idx;
    int* d_last_ref_idx;
    float* d_max_score;

    int h_last_query_idx;
    int h_last_ref_idx;
    float h_max_score;

    cudaMalloc((void**)&d_last_query_idx, sizeof(int));
    cudaMalloc((void**)&d_last_ref_idx, sizeof(int));
    cudaMalloc((void**)&d_max_score, sizeof(float));

    int offset = K / (n_block * n_thread);
    if (offset && K % (n_block * n_thread) != 0)
        offset += 1;
    offset = offset ? offset : 1;

    detect_cuda2<<<n_block, n_thread>>>(
        d_ref_index,
        d_ref_score,
        d_PN,
        L,
        K,
        tmp_wnd,
        offset,
        d_last_query_idx,
        d_last_ref_idx,
        d_max_score
    );


    cudaMemcpy(h_PN, d_PN, L * K * sizeof(PathNode), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_score, d_max_score, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_query_idx, d_last_query_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_ref_idx, d_last_ref_idx, sizeof(int), cudaMemcpyDeviceToHost);



#ifdef N_DEBUG
    printf("h_max_score : %f\n", h_max_score);
    printf("h_last_query_idx : %d\n", h_last_query_idx);
    printf("h_last_ref_idx : %d\n", h_last_ref_idx);
#endif

    //update result
    int p_i = h_last_query_idx;
    int p_j = h_last_ref_idx;

    int idx = 0;
    int match_cnt = 0;
    while(p_i != -1)
    {
#ifdef N_DEBUG
        printf("(%d %d)->", p_i, p_j);
#endif
        match_cnt++;
        result_score_path[idx] = ARR_I_J(_ref_score, p_i, p_j);
        res_q[idx] = p_i;
        res_p[idx++] = ARR_I_J(_ref_index, p_i, p_j);

        int c_i = ARR_I_J(h_PN, p_i, p_j).prevQueryIdx;
        int c_j = ARR_I_J(h_PN, p_i, p_j).prevTimestamp;
        p_i = c_i;
        p_j = c_j;
    }


    ////////////////////////////////////////////////////////////////////////////
    res->score = h_max_score;
    res->match = match_cnt;
    reverse_arr_f(result_score_path, res->match);
    reverse_arr(res_q, res->match);
    reverse_arr(res_p, res->match);
    ////////////////////////////////////////////////////////////////////////////


    cudaFree(d_ref_index);
    cudaFree(d_ref_score);
    cudaFree(d_PN);

#ifdef N_DEBUG
    printf("=== ARR_I_J(h_PN, i, j).score\n");
    for (int i = 0 ; i < L; i++)
    {
        for (int j = 0 ; j < K; j++)
        {
            printf("%f ", ARR_I_J(h_PN, i, j).score);
        }
        printf("\n");
    }

    printf("=====ARR_I_J(h_PN, i, j).prevTimestamp \n");
    for (int i = 0 ; i < L; i++)
    {
        for (int j = 0 ; j < K; j++)
        {
            printf("%d ", ARR_I_J(h_PN, i, j).prevTimestamp);
        }
        printf("\n");
    }

    printf("=====ARR_I_J(h_PN, i, j).prevQueryIdx \n");
    for (int i = 0 ; i < L; i++)
    {
        for (int j = 0 ; j < K; j++)
        {
            printf("%d ", ARR_I_J(h_PN, i, j).prevQueryIdx);
        }
        printf("\n");
    }
#endif
}

}