# Temporal Network CUDA imp

필요 파일

```bash
cuda/detect.cu
cuda/CudaFit.py  # python에서 cuda 실행
```
 
컴파일
 
 ```bash
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o cuda/detect.so --shared cuda/detect.cu 
```
 
사용법

- .cu 컴파일해서 .so 파일 생성
- .so파일은 한 번만 만들어놓으면 됩니다.
- `CudaFit` 클래스의 `__init__`에 `self.LIB_PATH`를 생성한 .so파일로 지정
