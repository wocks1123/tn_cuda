# Temporal Network CUDA imp

## 필요 파일

```bash
cuda/detect.cu
cuda/detect_vwii.cu
cuda/CudaFit.py
cuda/CudaFit_vwii.py
```



## 컴파일 

 ```bash
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda/detect.so --shared cuda/detect.cu
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda/detect_vwii.so --shared cuda/detect_vwii.cu 
 ```

- .cu 컴파일해서 .so 파일 생성
- `CudaFit` 클래스의 `__init__`에 `self.LIB_PATH`를 생성한 .so파일로 지정

```python
# 생성자에서
class CudaFit():
    def __init__(self, TOP_K=-1, SCORE_THR = 0.8, TEMP_WND=5, MIN_MATCH=5):
        # ...
        self.LIB_PATH = "cuda/detect.so" # <- 여기에 생성된 .so파일 경로 지정
```



## 사용법

- `main.py`에 실행 예제 코드 참고



기본 예시

```python
# CudaFit

from cuda.CudaFit import CudaFit

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
            'MIN_MATCH': 5}
c = CudaFit(**tn_param)
dt = c.detect_cuda(idx, score, 1, 64)
```

```python
# CudaFit_vwii

from cuda.CudaFit_vwii import CudaFit_vwii

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


tn_param = {'TOP_K': 64,
            'SCORE_THR': 0.7,
            'TEMP_WND': 10000,
            'MIN_MATCH': 5}
c = CudaFit_vwii(**tn_param)
ret = c.detect_cuda(idx, score, 1, 64)
```