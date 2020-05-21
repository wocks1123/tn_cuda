# coding=utf-8

import sys
import os
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from ctypes import *
import numpy as np
from utils.Period import Period


class CudaFit_vwii():
    def __init__(self, TOP_K=-1, SCORE_THR=0.8, TEMP_WND=5, MIN_MATCH=5):
        self.TOP_K = TOP_K              # -1 = all
        self.SCORE_THR = SCORE_THR
        self.MATCH_CNT = MIN_MATCH
        self.TMP_WND = TEMP_WND
        self.LIB_PATH = "cuda/detect_vwii.so"

    def detect_cuda(self, _ref_index, _ref_score, _n_block=1, _n_thread=64):
        """
        :param _ref_index:
        :param _ref_score:
        :return:
        """

        dll = CDLL(self.LIB_PATH)
        kernel_func = dll.call_kernel

        #################################################################################
        # set parameters type
        kernel_func.argtypes = [
            POINTER(c_int),     # ref_index(array 2D)
            POINTER(c_float),   # ref_score(array 2D)
            POINTER(c_int),     # video_idx(array 2D)
            c_int,              # L(query video length)
            c_int,              # K(ref video)
            c_float,            # score thr
            c_int,              # tmp window
            c_int,              # num_ref_video(len of idx_table)
            c_int,              # num of block
            c_int,              # num of thread
            ###################################
            POINTER(c_int),     # result query path(array 2D : K x L)
            POINTER(c_int),     # result ref path(array 2D : K x L)
            POINTER(c_float),   # result score path(array 2D : K x L)
            POINTER(c_int),     # result match count(array 1D : length L)
            POINTER(c_float)    # result match score(array 1D : length L)
        ]
        #################################################################################

        ref_index, ref_score, video_idx, K, L, idx_table = self.__get_parameters(_ref_index, _ref_score)


        # print("ref_index", ref_index)
        # print("ref_score", ref_score)
        # print("video_idx", video_idx)

        num_of_video = len(idx_table) # ref video 수

        #################################################################################
        # set parameters
        ref_index = np.array(ref_index).astype("int32")
        ref_score = np.array(ref_score).astype("float32")
        ref_score[ref_score < self.SCORE_THR] = 0
        video_idx = np.array(video_idx).astype("int32")

        result_query_path = np.full((num_of_video, L), -1).astype("int32")
        result_ref_path = np.full((num_of_video, L), -1).astype("int32")
        result_score_path = np.full((num_of_video, L), -1).astype("float32")
        result_match = np.full(num_of_video, 0).astype("int32")
        result_score = np.full(num_of_video, -1).astype("float32")

        p_ref_index = ref_index.ctypes.data_as(POINTER(c_int))
        p_ref_score = ref_score.ctypes.data_as(POINTER(c_float))
        p_video_idx = video_idx.ctypes.data_as(POINTER(c_int))

        p_result_query_path = result_query_path.ctypes.data_as(POINTER(c_int))
        p_result_ref_path = result_ref_path.ctypes.data_as(POINTER(c_int))
        p_result_score_path = result_score_path.ctypes.data_as(POINTER(c_float))
        p_result_match = result_match.ctypes.data_as(POINTER(c_int))
        p_result_score = result_score.ctypes.data_as(POINTER(c_float))
        #################################################################################


        kernel_func(
            p_ref_index,
            p_ref_score,
            p_video_idx,
            L,
            K,
            self.SCORE_THR,
            self.TMP_WND,
            num_of_video,
            _n_block,
            _n_thread,
            p_result_query_path,
            p_result_ref_path,
            p_result_score_path,
            p_result_match,
            p_result_score
        )


        # print("result_match", result_match)
        # print("result_ref_path", result_ref_path)
        # print("result_query_path", result_query_path)
        # print("result_score_path", result_score_path)
        # print("result_score", result_score)



        ret = []

        for i in range(len(result_ref_path)):
            if result_match[i] <= self.MATCH_CNT:
                continue

            query_index = result_ref_path[i][:result_match[i]]
            path = result_query_path[i][:result_match[i]]
            score_path = result_score_path[i][:result_match[i]]
            a = [
                idx_table[i],
                {
                    # "query_index": query_index,
                    # "score_path": score_path,
                    # "path": path,
                    "Query": Period(query_index[0], query_index[-1] + 1),
                    "Ref": Period(path[0], path[-1] + 1),
                    "match": result_match[i],
                    # "score": result_score[i]
                    "score": sum(score_path)

                }
            ]
            ret.append(a)

        return ret

    def __get_parameters(self, idx, score):
        """

        :param _ref_index:
        :param _ref_score:
        :return:
        """

        ref_index = []
        ref_score = []
        video_idx = []
        idx_table = []
        for i in range(len(idx)):
            ref_index_row = []
            ref_score_row = []
            video_idx_row = []
            for j in range(len(idx[i])):
                curr_vid_id, idx_list = idx[i][j]  # [video_idx, [timestamp, ...]]
                if curr_vid_id not in idx_table:
                    idx_table.append(curr_vid_id)
                for k in range(len(idx_list)):
                    ref_index_row.append(idx[i][j][1][k])
                    ref_score_row.append(score[i][j][1][k])
                    video_idx_row.append(idx_table.index(curr_vid_id))

            ref_index.append(ref_index_row)
            ref_score.append(ref_score_row)
            video_idx.append(video_idx_row)

        K = max([len(r) for r in ref_index])
        L = len(idx)

        for i in range(len(ref_index)):
            curr_row_len = len(ref_index[i])
            if curr_row_len < K:
                ref_index[i] = ref_index[i] + ([-1] * (K - curr_row_len))
                ref_score[i] = ref_score[i] + ([-1] * (K - curr_row_len))
                video_idx[i] = video_idx[i] + ([-1] * (K - curr_row_len))

        return ref_index, ref_score, video_idx, K, L, idx_table


    def test_foo(self, _ref_index, _ref_score):
        """
        for test...
        """
        ref_index, ref_score, video_idx, K, L, idx_table = self.__get_parameters(_ref_index, _ref_score)

        print("ref_index", ref_index)
        print("ref_score", ref_score)
        print("video_idx", video_idx)
        print("idx_table", idx_table)



        dll = CDLL("cuda/detect_vwii.so")
        kernel_func = dll.foo

        kernel_func.argtypes = [
            POINTER(c_int),     # ref_index(array 2D)
            c_int,              # L(query video length)
            c_int,              # K(ref video)
        ]

        ref_index = np.array(ref_index).astype("int32")
        ref_score = np.array(ref_score).astype("float32")
        p_ref_index = ref_index.ctypes.data_as(POINTER(c_int))

        kernel_func(
            p_ref_index,
            L,
            K
        )

        print("ref_index", ref_index)


if __name__ == "__main__":
    tn_param = {'TOP_K': 64,
                'SCORE_THR': 0.7,
                'TEMP_WND': 5,
                'MIN_MATCH': 5}

    idx = [
        [
            [1, [1, 2, 3]], [2, [2, 3, 4]], [4, [3, 4, 5]]
        ],
        [
            [1, [1, 2, 3]], [2, [2, 3, 4]], [4, [3, 4, 5]], [5, [6, 5, 7]]
        ],
        [
            [1, [1, 2, 3]], [2, [2, 3, 4]], [4, [3, 4, 5]]
        ]
    ]

    score = [
        [
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5]]
        ],
        [
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5]], [5, [0.5, 0.5, 0.5]]
        ],
        [
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5]]
        ]
    ]

    c = CudaFit_vwii(**tn_param)
    c.detect_cuda(idx, score)