"""
example code
"""


from cuda.CudaFit_vwii import CudaFit_vwii
from cuda.CudaFit import CudaFit


def example_cudafit():
    idx = [
        [0, 3, 2, 1],
        [1, 3, 2, 8],
        [2, 3, 4, 5],
        [3, 0, 4, 1]
    ]
    score = [
        [0.7, 0.3, 0.2, 0.1],
        [0.88, 0.83, 0.82, 0.80],
        [0.8, 0.41, 0.4, 0.3],
        [0.9, 0.8, 0.7, 0.51],
    ]

    tn_param = {'TOP_K': 64,
                'SCORE_THR': 0.7,
                'TEMP_WND': 5,
                'MIN_MATCH': 0}
    c = CudaFit(**tn_param)
    dt = c.detect_dp(idx, score, 1)
    print("dt", dt)

    dt = c.detect_cuda(idx, score, 1, 64)
    print("dt", dt)


def example_cudafit_vwii():
    tn_param = {'TOP_K': 20,
                'SCORE_THR': 0,
                'TEMP_WND': 10000,
                'MIN_MATCH': 0}

    idx = [
        [
            [1, [1, 2, 3]], [2, [2, 3, 4]], [4, [3, 4, 5, 6]], [7, [3, 4, 5, 6]]
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
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5, 0.6]], [7, [3, 4, 5, 6]]
        ],
        [
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5]], [5, [0.5, 0.5, 0.5]]
        ],
        [
            [1, [0.1, 0.2, 0.3]], [2, [0.2, 0.3, 0.4]], [4, [0.3, 0.4, 0.5]]
        ]
    ]

    c = CudaFit_vwii(**tn_param)
    ret = c.detect_cuda(idx, score, 1, 1)
    print("ret", ret)


if __name__ == "__main__":
    example_cudafit()

