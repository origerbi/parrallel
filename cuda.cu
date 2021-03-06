#include <cuda_runtime.h>
#include <limits.h>
#include <stdio.h>


__device__ const char *conservativeGroup[] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};
__device__ const char *semiConservativeGroup[] = {"SAG", "SGND", "NEQHRK", "ATV", "STPA", "NDEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM"};

typedef struct
{
    int n;
    int k;
}nkTuple;


// calculate string length
__host__ __device__ int strlenDevice(const char *str)
{
    int i = 0;
    while (str[i] != '\0')
    {
        i++;
    }
    return i;
}

__device__ int inString(const char *str, char c)
{
    for (int i = 0; str[i] != '\0'; i++)
    {
        if (str[i] == c)
        {
            return 0;
        }
    }
    return -1;
}

__device__ int compareInGroup(char a, char b, const char *group[], int groupLength)
{
    for (int i = 0; i < 9; i++)
    {
        if (inString(group[i], a) && inString(group[i], b))
            return 0;
    }
    return -1;
}

__device__ char getSign(char a, char b)
{
    if (a == b)
    {
        return '*';
    }
    switch (compareInGroup(a, b, conservativeGroup, 9))
    {
    case 0:
        return ':';
    case -1:
        switch (compareInGroup(a, b, semiConservativeGroup, 11))
        {
        case 0:
            return '.';
        default: // case -1: not in semiConservativeGroup
            return ' ';
        }
    }
    return ' ';
}

__device__ int calculateScore(char *seq1, char *seq2, int w[], int offset)
{
    int starsCount = 0;
    int colonsCount = 0;
    int pointsCount = 0;
    int spacesCount = 0;
    for (int i = 0; i < strlenDevice(seq2); i++)
    {
        switch (getSign(seq1[i + offset], seq2[i]))
        {
        case '*':
            starsCount++;
            break;
        case ':':
            colonsCount++;
            break;
        case '.':
            pointsCount++;
            break;
        case ' ':
            spacesCount++;
            break;
        }
    }
    return w[0] * starsCount - w[1] * colonsCount - w[2] * pointsCount - w[3] * spacesCount;
}

__device__ void getNKFromNumber(nkTuple *tuple,int number, int lengthSeq2)
{
    tuple->n = 0;
    tuple->k = 1;
    for(int i = 0; i < number; i++)
    {
        if (tuple->k < lengthSeq2 - 1) // increment k
            (tuple->k)++;
        else // increment n and reset k
        {
            (tuple->n)++;
            tuple->k = (tuple->n) + 1;
        }
    }

}

__global__ void BestMutantANDOffset(nkTuple* nk, char* seq1, char* seq2, int* w, int* offset,int* maxScore)           // each block represent mutant, each thread represent offset.
{
    __shared__ int bestOffset;
    __shared__ int bestScore;
    __shared__ char* seq2Shared;
    __shared__ nkTuple myTuple;
    *maxScore = INT_MIN;

    if(threadIdx.x == 0)
    {
        bestOffset = 0;
        bestScore = INT_MIN;
        int len =  strlenDevice(seq2);
        
        seq2Shared = (char*)malloc(len-1);
        
        seq2Shared[strlenDevice(seq2) - 2] = '\0';
                
        getNKFromNumber(&myTuple,blockIdx.x, strlenDevice(seq2));
        int counter = 0;

        for (int j = 0; j < strlenDevice(seq2); j++) // generate mutant
        {
            if (j != myTuple.n && j != myTuple.k) // if not equal to n and k
            {
                seq2Shared[counter] = seq2[j];
                counter++;
            }
        }
    }

    __syncthreads();
    int score = calculateScore(seq1, seq2Shared, w, threadIdx.x);
    atomicMax(&bestScore, score);
    __syncthreads();

    if(bestScore == score)
    {
        bestOffset = threadIdx.x;
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        atomicMax(maxScore, bestScore);
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        if(bestScore == *maxScore)
        {
            nk->n = myTuple.n;
            nk->k = myTuple.k;
            *offset = bestOffset;
        }
    free(seq2Shared);
    }
}

extern "C" int* getBestMutantCuda(char *seq1, char *seq2, int w[])
{
    int* result = (int*)malloc(sizeof(int) * 3);
    char* seq1Device;
    cudaMalloc(&seq1Device, sizeof(char) * strlenDevice(seq1) + 1);
    cudaMemcpy(seq1Device, seq1, sizeof(char) * strlenDevice(seq1) + 1, cudaMemcpyHostToDevice);
    char* seq2Device;
    int* weightsDevice;
    cudaMalloc(&seq2Device, sizeof(char) * strlenDevice(seq2) + 1);
    cudaMalloc(&weightsDevice, sizeof(int) * 4);
    cudaMemcpy(seq2Device, seq2, sizeof(char) * strlenDevice(seq2) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(weightsDevice, w, sizeof(int) * 4, cudaMemcpyHostToDevice);
    nkTuple* nkDevice;
    int* offsetDevice;
    nkTuple* nk = (nkTuple*)malloc(sizeof(nkTuple));
    cudaMalloc(&nkDevice, sizeof(nkTuple));
    cudaMalloc(&offsetDevice, sizeof(int));
    int n = strlen(seq2) - 1;
    int s = n * (1 + n) / 2; // sum of all possible mutants
    int* maxScore;
    cudaMalloc(&maxScore,sizeof(int));
    int* helper = (int*)malloc(sizeof(int));
    *helper = 999;
    BestMutantANDOffset<<<s,strlen(seq1)-strlen(seq2) + 3>>>(nkDevice,seq1Device,seq2Device,weightsDevice,offsetDevice,maxScore);
    cudaMemcpy(helper,maxScore,sizeof(int),cudaMemcpyDeviceToHost);
    printf("max Score (helper): %d\n",*helper);
    cudaMemcpy(result,offsetDevice,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(nk,nkDevice,sizeof(nkTuple),cudaMemcpyDeviceToHost);
    result[2] = nk->k;
    result[1] = nk->n;

    printf("%d ,%d ,%d\n",result[0],result[1],result[2]);
    free(nk);
    
    cudaFree(seq1Device);
    cudaFree(seq2Device);
    cudaFree(weightsDevice);
    cudaFree(nkDevice);
    cudaFree(maxScore);
    cudaFree(offsetDevice);
    return result;
}
