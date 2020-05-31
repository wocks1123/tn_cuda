#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <cstdlib>

#define ARR_I_J(arr, i, j) arr[(i) * (K) + (j)]
#define ARR_I_J_W(arr, i, j, W) arr[(i) * (L) + (j)]


//#define N_DEBUG

struct PathNode
{
    int prevTimestamp;
    int prevQueryIdx;
    int videoId;
    float score;

public:
    PathNode() : prevTimestamp(-1), prevQueryIdx(-1), videoId(-1), score(0)
    {
    }

    PathNode(int _videoId, float _score) : prevTimestamp(-1), prevQueryIdx(-1), videoId(_videoId), score(_score)
    {
    }
};

__global__ void detect_cuda_vwii(const int* d_ref_index, const float* d_ref_score, int* const d_video_idx, PathNode* d_PN, const int L, const int K, int tmp_wnd, const int offset,
                           int* d_last_timestamp_list, int* d_last_queryidx_list, float* d_res_score_list
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
            int curr_videoidx = ARR_I_J(d_video_idx, i, j);
            if (curr_videoidx == -1)
            {
                continue;
            }

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
                    int prev_videoidx = ARR_I_J(d_video_idx, l, k);
                    if (curr_videoidx != prev_videoidx)
                    {
                        continue;
                    }
                    if (prev_videoidx == -1)
                    {
                        continue;
                    }


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
            ARR_I_J(d_PN, i, j).videoId = curr_videoidx;
            ARR_I_J(d_PN, i, j).score = max_score;

            if (d_res_score_list[curr_videoidx] <= max_score)
            {
                d_last_queryidx_list[curr_videoidx] = i;
                d_last_timestamp_list[curr_videoidx] = j;
                d_res_score_list[curr_videoidx] = max_score;
            }

        }
        __syncthreads();
    }

#ifdef N_DEBUG
    printf("==d_last_queryidx_list\n");
    for (int j = 0 ; j < K; j++)
    {
        printf("%d ", d_last_queryidx_list[j]);
    }
    printf("\n\n");

    printf("==d_last_timestamp_list\n");
    for (int j = 0 ; j < K; j++)
    {
        printf("%d ", d_last_timestamp_list[j]);
    }
    printf("\n\n");

    printf("==d_res_score_list\n");
    for (int j = 0 ; j < K; j++)
    {
        printf("%f ", d_res_score_list[j]);
    }
    printf("\n\n");
#endif


}



__global__ void update_result(const int* d_ref_index, const float* d_ref_score, int* const d_video_idx, PathNode* d_PN, const int L, const int K, const int video_num,
                                 int* d_last_timestamp_list, int* d_last_queryidx_list, float* d_res_score_list, const int offset, int* d_res_q, int* d_res_p, float* d_res_scores, int* d_match
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int s_idx = tid * offset;
    int e_idx = s_idx + offset;
    if (e_idx > video_num)
        e_idx = video_num;

#ifdef N_DEBUG
    printf("[tid:%d]update_result %d ~ %d(%d) %d\n", tid, s_idx, e_idx, offset , video_num);
#endif

    if (tid >= K)
    {
        return;
    }

    for(int t = s_idx; t < e_idx; t++)
    {


        int last_timestamp = d_last_timestamp_list[t];
        int last_queryidx = d_last_queryidx_list[t];
        if (last_queryidx == -1) continue;

        //printf("[%d](%d %d),\n", t, last_queryidx, last_timestamp);

        int p_i = last_queryidx;
        int p_j = last_timestamp;

        int idx = 0;
        int match_cnt = 0;
        while(p_i != -1)
        {
            //printf("(%d %d)->", p_i, p_j);
            match_cnt++;
            ARR_I_J_W(d_res_scores, t, idx, L) = ARR_I_J(d_ref_score, p_i, p_j);
            ARR_I_J_W(d_res_q, t, idx, L) = p_i;
            ARR_I_J_W(d_res_p, t, idx++, L) = ARR_I_J(d_ref_index, p_i, p_j);

            int c_i = ARR_I_J(d_PN, p_i, p_j).prevQueryIdx;
            int c_j = ARR_I_J(d_PN, p_i, p_j).prevTimestamp;
            p_i = c_i;
            p_j = c_j;
        }
        //printf("\n");


        //reverse array
        for(int i = 0; i < match_cnt / 2; i++)
        {
            int temp = ARR_I_J_W(d_res_q, t, match_cnt - 1 - i, video_num);
            ARR_I_J_W(d_res_q, t, match_cnt - 1 - i, video_num) = ARR_I_J_W(d_res_q, t, i, video_num);
            ARR_I_J_W(d_res_q, t, i, video_num) = temp;

            temp = ARR_I_J_W(d_res_p, t, match_cnt - 1 - i, video_num);
            ARR_I_J_W(d_res_p, t, match_cnt - 1 - i, video_num) = ARR_I_J_W(d_res_p, t, i, video_num);
            ARR_I_J_W(d_res_p, t, i, video_num) = temp;

            float f_temp = ARR_I_J_W(d_res_scores, t, match_cnt - 1 - i, video_num);
            ARR_I_J_W(d_res_scores, t, match_cnt - 1 - i, video_num) = ARR_I_J_W(d_res_scores, t, i, video_num);
            ARR_I_J_W(d_res_scores, t, i, video_num) = f_temp;
        }

        d_match[t] = match_cnt;
    }

    __syncthreads();
}


#ifdef __cplusplus
extern "C" {
#endif

void call_kernel(int* _ref_index, float* _ref_score, int* _video_idx, int L, int K, float score_thr, int tmp_wnd, int video_num, int n_block, int n_thread,
                int* res_q, int* res_p, float* result_score_path, int* match, float* score)
{

    //printf("call_kernel called! L : %d K : %d vn : %d\n",L, K, video_num);
    int* h_listidx_list = (int*)calloc(K, sizeof(int));
    int* h_queryidx_list = (int*)calloc(K, sizeof(int));
    float* h_maxscore_list = (float*)calloc(K, sizeof(float));

    memset(h_listidx_list, -1, K * sizeof(int));
    memset(h_queryidx_list, -1, K * sizeof(int));

    int* d_ref_index;
    float* d_ref_score;
    int* d_video_idx;
    PathNode* d_PN;
    int* d_lastidx_list;
    int* d_last_queryidx_list;
    float* d_maxscore_list;

    int* d_res_q;
    int* d_res_p;
    float* d_res_scores;
    int* d_match;

    cudaMalloc((void**)&d_ref_index, L * K * sizeof(int));
    cudaMalloc((void**)&d_ref_score, L * K * sizeof(float));
    cudaMalloc((void**)&d_video_idx, L * K * sizeof(int));
    cudaMalloc((void**)&d_PN, L * K * sizeof(PathNode));
    cudaMalloc((void**)&d_lastidx_list, video_num * sizeof(int));
    cudaMalloc((void**)&d_last_queryidx_list, video_num * sizeof(int));
    cudaMalloc((void**)&d_maxscore_list, video_num * sizeof(float));
    cudaMalloc((void**)&d_res_q, L * video_num * sizeof(int));
    cudaMalloc((void**)&d_res_p, L * video_num * sizeof(int));
    cudaMalloc((void**)&d_res_scores, L * video_num * sizeof(float));
    cudaMalloc((void**)&d_match, video_num * sizeof(int));


    /*
    *   set variables
    */
    PathNode* h_PN = new PathNode[L * K];
    //printf("update first row\n");
    for (int i = 0; i < K; i++)
    {
        ARR_I_J(h_PN, 0, i).videoId = ARR_I_J(_video_idx, 0, i);
        ARR_I_J(h_PN, 0, i).score = ARR_I_J(_ref_score, 0, i);
        //printf("%f ", ARR_I_J(_ref_score, 0, i));
    }
    //printf("\n");

#ifdef N_DEBUG
    printf("K : %d\n", K);
    printf("vidnum : %d\n", video_num);
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

    cudaMemcpy(d_lastidx_list, h_listidx_list, K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_last_queryidx_list, h_queryidx_list, K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxscore_list, h_maxscore_list, K * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_ref_index, _ref_index, L * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_score, _ref_score, L * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_video_idx, _video_idx, L * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PN, h_PN, L * K * sizeof(PathNode), cudaMemcpyHostToDevice);

    cudaMemcpy(d_res_q, res_q, L * video_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_p, res_p, L * video_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_scores, result_score_path, L * video_num * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_match, match, video_num * sizeof(int), cudaMemcpyHostToDevice);

    int offset = K / (n_block * n_thread);
    if (offset && K % (n_block * n_thread) != 0)
        offset += 1;
    offset = offset ? offset : 1;

    cudaDeviceSynchronize();

    detect_cuda_vwii<<<n_block, n_thread>>>(
        d_ref_index,
        d_ref_score,
        d_video_idx,
        d_PN,
        L,
        K,
        tmp_wnd,
        offset,
        d_lastidx_list,
        d_last_queryidx_list,
        d_maxscore_list
    );

    offset = video_num / (n_block * n_thread);
    if (offset && video_num % (n_block * n_thread) != 0)
        offset += 1;

    offset = offset ? offset : 1;
    update_result<<<n_block, n_thread>>>(
        d_ref_index,
        d_ref_score,
        d_video_idx,
        d_PN,
        L,
        K,
        video_num,
        d_lastidx_list,
        d_last_queryidx_list,
        d_maxscore_list,
        offset,
        d_res_q,
        d_res_p,
        d_res_scores,
        d_match
    );

    //cudaMemcpy(_ref_index, d_ref_index, L * K * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(_ref_score, d_ref_score, L * K * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(_video_idx, d_video_idx, L * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PN, d_PN, L * K * sizeof(PathNode), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_q, d_res_q, L * video_num * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_p, d_res_p, L * video_num * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result_score_path, d_res_scores, L * video_num * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(score, d_maxscore_list, video_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(match, d_match, video_num * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ref_index);
    cudaFree(d_ref_score);
    cudaFree(d_video_idx);
    cudaFree(d_PN);
    cudaFree(d_lastidx_list);
    cudaFree(d_last_queryidx_list);
    cudaFree(d_maxscore_list);
    cudaFree(d_res_q);
    cudaFree(d_res_p);
    cudaFree(d_res_scores);
    cudaFree(d_match);

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
    printf("===match\n");
    for (int j = 0 ; j < video_num; j++)
    {
        printf("%d(%d) ", match[j], j);
    }
    printf("\n");
    printf("===res_p\n");
    for (int i = 0 ; i < video_num; i++)
    {
        for (int j = 0 ; j < match[i]; j++)
        {
            printf("%d ", ARR_I_J_W(res_p, i, j, video_num));
        }
        printf("\n");
    }
#endif
}

void foo(int* arr2d, int L, int K)
{
    printf("fun called!\n");

    for(int i = 0 ; i < L; i++)
    {
        for (int j = 0 ; j < K; j++)
        {
            ARR_I_J(arr2d, i, j) = 100;
            printf("%d ", ARR_I_J(arr2d, i, j));

        }
        printf("\n");
    }
}


#ifdef __cplusplus
}
#endif