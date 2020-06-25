# coding=utf-8

import sys
import os
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import numpy as np
import ctypes
from ctypes import *
from utils.Period import Period
import copy


class CudaFit():
    def __init__(self, TOP_K=-1, SCORE_THR = 0.8, TEMP_WND=5, MIN_MATCH=5):
        self.TOP_K = TOP_K
        self.SCORE_THR = SCORE_THR
        self.MATCH_CNT = MIN_MATCH
        self.TMP_WND = TEMP_WND
        self.LIB_PATH = "cuda/detect.so"

    def detect_dp(self, _ref_index, _ref_score, N):
        """
        - 조건 만족하는 모든 path, score 반환
        """
        L = len(_ref_index)

        ref_index = np.array(_ref_index)
        ref_score = np.array(_ref_score)

        ref_score[ref_score < self.SCORE_THR] = 0

        class NodeInfo():
            def __init__(self, path, score, query_list):
                self.path = path # list
                self.score = score # float
                self.query_list = query_list # list
            def __repr__(self):
                return "(path {} query {} score {})".format(self.path, self.query_list, self.score)

        prev_indexs = [[NodeInfo([ref_index[j][i]], score=ref_score[j][i], query_list=[j]) for i in range(len(ref_index[j]))] for j in
                       range(L)]

        search_cnt = 0

        candidates = []
        for i in range(1, L):
            curr_indexs = [NodeInfo([_i], score=ref_score[i][_i], query_list=[i]) for _i in range(len(ref_index[i]))]
            for j in range(len(ref_index[i])):
                curr_index = ref_index[i][j]
                tmp_wnd = i - self.TMP_WND if i - self.TMP_WND >= 0 else 0
                for m in range(tmp_wnd, i): # 0 -> i // 0 -> curr L - 1
                    for prev in prev_indexs[m]: # 0 -> K
                        search_cnt += 1
                        _paths, _score, _query_idx = prev.path, prev.score, prev.query_list
                        if _score <= 0:
                            continue
                        last_index = _paths[-1]
                        if curr_index - self.TMP_WND < last_index < curr_index:
                            if ref_score[i][j] <= 0:
                                continue
                            if curr_indexs[j].score < _score + ref_score[i][j]:
                                updated_node = NodeInfo(_paths + [curr_index], _score + ref_score[i][j], _query_idx + [i])
                                curr_indexs[j] = updated_node
                                if len(updated_node.path) >= self.MATCH_CNT:
                                    candidates.append(updated_node)

            prev_indexs[i] = curr_indexs

        if len(prev_indexs) == 0:
            return []

        candidates.sort(key=lambda _x: _x.score, reverse=True)

        ret = [{
            "Ref": Period(node_info.path[0], node_info.path[-1] + 1),
            "ref_index": node_info.path,
            "Query": Period(node_info.query_list[0], node_info.query_list[-1] + 1),
            "query_index": node_info.query_list,
            "match": len(node_info.query_list),
            "score": node_info.score,
            "search_cnt": search_cnt,
        } for node_info in candidates][:N]
        return ret


    # def detect_cuda(self, _ref_index, _ref_score, _n_block=1, _n_thread=64):
    #     L = len(_ref_index)
    #     K = self.TOP_K if self.TOP_K < len(_ref_index[0]) else len(_ref_index[0])
    #
    #     ref_index = copy.deepcopy(_ref_index)
    #     ref_score = copy.deepcopy(_ref_score)
    #
    #     if K != len(_ref_index[0]):
    #         ref_index = np.array(_ref_index)[:, :K]
    #         ref_score = np.array(_ref_score)[:, :K]
    #
    #     for i in range(0, L):
    #         for j in range(0, K):
    #             if ref_score[i][j] < self.SCORE_THR:
    #                 ref_score[i][j] = 0
    #
    #     class RESULT(Structure):
    #         _fields_ = [("score", c_float), ("match", c_int)]
    #
    #     dll = CDLL(self.LIB_PATH)
    #     kernel_func = dll.call_kernel
    #     kernel_func.argtypes = [
    #         POINTER(c_int),
    #         POINTER(c_float),
    #         c_size_t,
    #         c_size_t,
    #         c_float,
    #         c_int,
    #         POINTER(c_int),
    #         POINTER(c_int),
    #         c_int,
    #         c_int,
    #         POINTER(RESULT)
    #     ]
    #
    #     ref_index = np.array(ref_index).astype("int32")
    #     ref_score = np.array(ref_score).astype("float32")
    #     query_index = np.full(L, -1).astype("int32")
    #     path = np.full(L, -1).astype("int32")
    #     result = RESULT()
    #
    #     p_ref_index = ref_index.ctypes.data_as(POINTER(c_int))
    #     p_ref_score = ref_score.ctypes.data_as(POINTER(c_float))
    #     p_query_index = query_index.ctypes.data_as(POINTER(c_int))
    #     p_path = path.ctypes.data_as(POINTER(c_int))
    #     p_result = ctypes.byref(result)
    #
    #     kernel_func(p_ref_index, p_ref_score, L, K, self.SCORE_THR, self.TMP_WND, p_query_index, p_path, _n_block, _n_thread, p_result)
    #
    #     path = path[:result.match]
    #     query_index = query_index[:result.match]
    #
    #     if result.match <= self.MATCH_CNT:
    #         return []
    #
    #     if result.score == 0:
    #         return []
    #
    #     # print("query_index", query_index)
    #     # print("max_path", path)
    #     # print("max_weight", result.score)
    #
    #     dict = {
    #         "Query" : Period(query_index[0], query_index[-1] + 1),
    #         "query_index": query_index,
    #         "Ref" : Period(path[0], path[-1] + 1),
    #         "ref_index": path,
    #         "match" : result.match,
    #         "score" : result.score
    #     }
    #
    #     return [dict]

    def detect_cuda(self, _ref_index, _ref_score, _n_block=1, _n_thread=64):
        L = len(_ref_index)
        K = self.TOP_K if self.TOP_K < len(_ref_index[0]) else len(_ref_index[0])

        ref_index = copy.deepcopy(_ref_index)
        ref_score = copy.deepcopy(_ref_score)

        if K != len(_ref_index[0]):
            ref_index = np.array(_ref_index)[:, :K]
            ref_score = np.array(_ref_score)[:, :K]

        for i in range(0, L):
            for j in range(0, K):
                if ref_score[i][j] < self.SCORE_THR:
                    ref_score[i][j] = 0

        class RESULT(Structure):
            _fields_ = [("score", c_float), ("match", c_int)]

        dll = CDLL(self.LIB_PATH)
        kernel_func = dll.call_kernel2
        kernel_func.argtypes = [
            POINTER(c_int),     # idx
            POINTER(c_float),   # score
            c_int,      # L
            c_int,      # K
            c_float,    # score_thr
            c_int,      # temp_wnd
            c_int,      # n_block
            c_int,      # n_thread
            POINTER(c_int), # res query
            POINTER(c_int),  # res path
            POINTER(c_float),  # res score path
            POINTER(RESULT)
        ]

        ref_index = np.array(ref_index).astype("int32")
        ref_score = np.array(ref_score).astype("float32")
        query_index = np.full(L, -1).astype("int32")
        path = np.full(L, -1).astype("int32")
        score_path = np.full(L, -1).astype("float32")
        result = RESULT()

        p_ref_index = ref_index.ctypes.data_as(POINTER(c_int))
        p_ref_score = ref_score.ctypes.data_as(POINTER(c_float))
        p_query_index = query_index.ctypes.data_as(POINTER(c_int))
        p_path = path.ctypes.data_as(POINTER(c_int))
        p_score_path = score_path.ctypes.data_as(POINTER(c_float))
        p_result = ctypes.byref(result)

        kernel_func(
            p_ref_index,
            p_ref_score,
            L,
            K,
            self.SCORE_THR,
            self.TMP_WND,
            _n_block,
            _n_thread,
            p_query_index,
            p_path,
            p_score_path,
            p_result
        )

        path = path[:result.match]
        query_index = query_index[:result.match]

        # print("path", path)
        # print("query_index", query_index)

        if result.match <= self.MATCH_CNT:
            return []

        if result.score == 0:
            return []

        # print("query_index", query_index)
        # print("max_path", path)
        # print("max_weight", result.score)

        dict = {
            "Query" : Period(query_index[0], query_index[-1] + 1),
            "query_index": query_index,
            "Ref" : Period(path[0], path[-1] + 1),
            "ref_index": path,
            "match" : result.match,
            "score" : result.score
        }

        return [dict]

    # @DEPRECATED
    # cuda 코드랑 비교용
    #
    # def detect_dp(self, _ref_index, _ref_score):
    #     """
    #     - path 검색
    #
    #     :param ref_index
    #         - 2차원 array
    #         - 각 행은 query의 index에 매칭된 ref video의 timestamp 목록
    #     :param ref_score
    #         - 2차원 array
    #         - 각 행은 해당 query와 유사도
    #     :param L
    #         - query video 길이
    #     :param K
    #         - top-k로 가져올 개수
    #
    #     :return
    #         - (query_index, path, match, score)
    #             - query_index : path와 매칭된 query의 timestamp
    #             - path : 최고점을 가진 ref video의 path
    #             - match : path 길이
    #             - score : 최고점
    #     """
    #     L = len(_ref_index)
    #     #K = self.TOP_K if self.TOP_K < len(_ref_index[0]) else len(_ref_index[0])
    #
    #     # if K != len(_ref_index[0]):
    #     #     ref_index = np.array(_ref_index)[:, :K]
    #     #     ref_score = np.array(_ref_score)[:, :K]
    #     # else:
    #     #     ref_index = np.array(_ref_index)
    #     #     ref_score = np.array(_ref_score)
    #
    #     ref_index = np.array(_ref_index)
    #     ref_score = np.array(_ref_score)
    #     score = [[0 for j in range(len(ref_score[i]))] for i in range(L)]
    #     index = [[(-1, -1) for j in range(len(ref_index[i]))] for i in range(L)]  # (prev row, prev_frame_idx)
    #
    #     for i in range(0, L):
    #         #for j in range(0, K):
    #         for j in range(0, len(ref_score[i])): # K
    #             if ref_score[i][j] < self.SCORE_THR:
    #                 ref_score[i][j] = 0
    #
    #     #for i in range(K):
    #     for i in range(len(_ref_score[0])): # K
    #         score[0][i] = ref_score[0][i]
    #
    #     ret_score = -1
    #     res_index = []
    #     last_index = -1
    #     last_i = -1
    #     search_cnt = 0
    #     for i in range(0, L):
    #         #for j in range(0, K):
    #         for j in range(0, len(ref_score[i])): # K
    #             index[i][j] = (-1, -1)
    #             curr_score = ref_score[i][j]
    #             curr_index = ref_index[i][j]
    #
    #             if curr_score < self.SCORE_THR:
    #                 continue
    #
    #             max_score = curr_score
    #             prev_i = -1
    #             prev_j = -1
    #
    #             for l in range(0, i): # l => 0 ~ i - 1
    #                 # for k in range(0, K):
    #                 for k in range(0, len(ref_score[i])):  # K
    #                     prev_score = score[l][k]
    #                     prev_index = ref_index[l][k]
    #                     search_cnt += 1
    #                     if prev_index >= curr_index:
    #                         continue
    #
    #                     if max_score < prev_score + curr_score:
    #                         max_score = prev_score + curr_score
    #                         prev_i = l
    #                         prev_j = k
    #
    #             score[i][j] = max_score
    #             index[i][j] = (prev_i, prev_j)
    #
    #             if ret_score < max_score:
    #                 ret_score = max_score
    #                 res_index = (prev_i, prev_j)
    #                 last_index = ref_index[i][j]
    #                 last_i = i
    #
    #     if last_index == -1:
    #         #print("not matched")
    #         return []
    #
    #     res_path = []
    #     query_index = []
    #     p_i = res_index[0]
    #     p_j = res_index[1]
    #     res_path.append(last_index)
    #     query_index.append(last_i)
    #
    #     cnt = 1
    #     while p_i != -1:
    #         val = index[p_i][p_j]
    #         res_path.append(ref_index[p_i][p_j])
    #         query_index.append(p_i)
    #         p_i = val[0]
    #         p_j = val[1]
    #         cnt += 1
    #
    #     res_path.reverse()
    #     query_index.reverse()
    #
    #     if cnt <= self.MATCH_CNT:
    #         return []
    #
    #     # print("query_index", query_index)
    #     # print("max_path", res_path)
    #     # print("max_weight", ret_score)
    #
    #     dict = {
    #         "query": Period(query_index[0], query_index[-1] + 1),
    #         "ref": Period(res_path[0], res_path[-1] + 1),
    #         "match": cnt,
    #         "score": ret_score,
    #         "search_cnt": search_cnt,
    #         "query frame index": query_index,
    #         "detect path": res_path
    #         #"scores": max_score_path
    #     }
    #
    #     #return query_index, res_path, cnt, ret_score
    #     return [dict]

if __name__ == "__main__":
    tn_param = {'TOP_K': 64,
                'SCORE_THR': 0.7,
                'TEMP_WND': 5,
                'MIN_MATCH': 5}

    idx = [
        [
            [1, [1,2,3]], [2,[2,3,4]], [4,[3,4,5]]
        ],
        [
            [1, [1, 2, 3]], [2, [2, 3, 4]], [4, [3, 4, 5]], [5,[6,5,7]]
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


    c = CudaFit(**tn_param)
    print("hjello")
