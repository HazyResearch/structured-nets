/*
Copyright 2018 HazyResearch
https://github.com/HazyResearch/structured-nets

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

__global__ void subdiagMult(
    float* d_Subdiag,
    float* d_Data,
    float* d_Output,
    int shiftSubdiag,
    int shiftV,
    int N,
    int subdiagOffset) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;

  float* d_Src = d_Data + blockIdx.y * N;
  float* d_Dst = d_Output + blockIdx.y * N;
  float* d_Sub = d_Subdiag + blockIdx.y * subdiagOffset;
  // for (int pos = tid; pos < N; pos += numThreads) {
  if (pos < N) {
    d_Dst[pos] =
        d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N];
  }
}

void subdiagMultGPU(
    float* d_Subdiag,
    float* d_Data,
    float* d_Output,
    int shiftSubdiag,
    int shiftV,
    int batchSize,
    int N,
    bool batchedSubdiag) {
  const int THREAD_N = 256;
  dim3 grid((N + THREAD_N - 1) / THREAD_N, batchSize);
  int subdiagOffset = batchedSubdiag ? N : 0;
  subdiagMult<<<grid, THREAD_N>>>(
      d_Subdiag, d_Data, d_Output, shiftSubdiag, shiftV, N, subdiagOffset);
}
