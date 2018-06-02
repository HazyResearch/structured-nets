__global__ void subdiagMult(float *d_Subdiag, float *d_Data, float *d_Output, int shiftSubdiag, int shiftV, int N, int subdiagOffset) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;

    float *d_Src = d_Data  + blockIdx.y * N;
    float *d_Dst = d_Output + blockIdx.y * N;
    float *d_Sub = d_Subdiag + blockIdx.y * subdiagOffset;
    // for (int pos = tid; pos < N; pos += numThreads) {
    if (pos < N) {
        d_Dst[pos] = d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N];
    }
}

void subdiagMultGPU(float *d_Subdiag, float *d_Data, float *d_Output, int shiftSubdiag, int shiftV, int batchSize, int N, bool batchedSubdiag) {
    const int THREAD_N = 256;
    dim3 grid((N + THREAD_N - 1) / THREAD_N, batchSize);
    int subdiagOffset = batchedSubdiag ? N : 0;
    subdiagMult<<<grid, THREAD_N>>>(d_Subdiag, d_Data, d_Output, shiftSubdiag, shiftV, N, subdiagOffset);
}