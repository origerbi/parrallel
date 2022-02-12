#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#define input_file "input.txt"
#define output_file "output.txt"

const char *conservativeGroup[] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};
const char *semiConservativeGroup[] = {"SAG", "SGND", "NEQHRK", "ATV", "STPA", "NDEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM"};
const char sign[4] = {'*', ' ', ':', '.'};
enum tags {
    WORK,
    STOP,
    RESULT
};

typedef struct
{
    int w[4];
    char *seq1;
    int seq2Count;
    char **seq2;
} Input;

typedef struct
{
    int n;
    int k;
    int bestOffset;
} Mutant;

typedef struct 
{
    int lineNum;
    char* data;
    Mutant mutant;
}output_fileStruct;


extern int* getBestMutantCuda(char *seq1, char *seq2, int w[]);
int compare(char *a, char *b);
int inString(const char *str, char c);
int compareInGroup(char a, char b, const char *group[], int groupLength);
int calculateScore(char *seq1, char *seq2, int w[], int offset);
char getSign(char a, char b);
int calcaulateBestAligment(char *seq1, char *seq2, int w[], int *offset);
Mutant getBestMutant(char *seq1, char *seq2, int w[]);
Input *readFromFile();
void writeToFile(output_fileStruct *outputList,int seq2Count);
void master(Input *input, int world_size);
void slave(int rank);
void freeFunc(Input *input);


// compare two charcters
int compare(char *a, char *b)
{
    return *a - *b;
}

// check if char in string
int inString(const char *str, char c)
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

// compare two charcters under same string at group
int compareInGroup(char a, char b, const char *group[], int groupLength)
{
    for (int i = 0; i < groupLength; i++)
    {
        if (inString(group[i], a) && inString(group[i], b))
            return 0;
    }
    return -1;
}

char getSign(char a, char b)
{
    if (a==b)
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
}

int calculateScore(char *seq1, char *seq2, int w[], int offset)
{
    int starsCount = 0;
    int colonsCount = 0;
    int pointsCount = 0;
    int spacesCount = 0;
    for (int i = 0; i < strlen(seq2); i++)
    {
        switch (getSign(seq1[i+offset], seq2[i]))
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

int calcaulateBestAligment(char *seq1, char *seq2, int w[], int *offset)
{
    int bestScore = INT_MIN;
    int bestOffset = 0;
    for (int i = 0; i < strlen(seq2); i++)
    {
        int score = calculateScore(seq1, seq2, w, i);
        if (score > bestScore)
        {
            bestScore = score;
            bestOffset = i;
        }
    }
    *offset = bestOffset;
    return bestScore;
}

Mutant getBestMutant(char *seq1, char *seq2, int w[])
{
    Mutant bestMutant;
    int n = strlen(seq2) - 1;
    int s = n * (1 + n) / 2; // sum of all possible mutants
    n = 0;
    int k = 1;
    char *helper;
    int bestScoreMutant = INT_MIN;
#pragma omp parallel for private(helper)
    for (int i = 0; i < s; i++)
    {
        int counter = 0;
        helper = (char *)malloc(sizeof(char) * strlen(seq2) - 1);
        helper[strlen(seq2) - 2] = '\0';
        for (int j = 0; j < strlen(seq2); j++) // generate mutant
        {
            if (j != n && j != k) // if not equal to n and k
            {
                helper[counter] = seq2[j];
                counter++;
            }
        }
        int offset = 0;
        int score = calcaulateBestAligment(seq1, helper, w, &offset); // calculate best aligment
        free(helper);
#pragma omp critical // critical section. to not override the best score
        {
            if (score > bestScoreMutant)
            {
                bestScoreMutant = score;
                bestMutant.n = n;
                bestMutant.k = k;
                bestMutant.bestOffset = offset;
            }
        }
        if (k < strlen(seq2) - 1) // increment k
            k++;
        else // increment n and reset k
        {
            n++;
            k = n + 1;
        }
    }
    return bestMutant;
}

Input *readFromFile()
{
    FILE *fp;
    Input *input;
    input = malloc(sizeof(Input));
    input->seq1 = (char *)malloc(sizeof(char) * 5001);
    input->seq2 = (char **)malloc(sizeof(char *) * 5001);
    // open file
    fp = fopen(input_file, "r");
    if (fp == NULL)
    {
        printf("Error opening file!\n");
    }
    fscanf(fp, "%d, %d, %d, %d\n", input->w, input->w+1, input->w+2, input->w+3); // read weights
    size_t temp;
    getline(&input->seq1, &temp, fp);                                                    // read seq1
    fscanf(fp, "%d \n", &input->seq2Count);                                               // read seq2Count
    for (int i = 0; i < input->seq2Count; i++)                                          // read seq2
    {
        input->seq2[i] = (char *)malloc(sizeof(char) * 5001);
        temp = getline(&input->seq2[i], &temp, fp);
        input->seq2[i][temp-1] = '\0'; 
    }
    // close file
    fclose(fp);
    return input;
}

void writeToFile(output_fileStruct *outputList,int seq2Count)
{
    FILE *fp;
    fp = fopen(output_file, "w");
    if (fp == NULL)
    {
        printf("Error opening file!\n");
    }
    for(int i=0; i< seq2Count; i++)
    {
        fprintf(fp,"%s %d %d %d\n",outputList[i].data,outputList[i].mutant.bestOffset,outputList[i].mutant.n,outputList[i].mutant.k);
    }
    fclose(fp);
}

void freeFunc(Input *input)
{
    free(input->seq1);
    for (int i = 0; i < input->seq2Count; i++)
        free(input->seq2[i]);
    free(input->seq2);
    free(input);
}

void master(Input *input, int world_size)
{
        int sentWork = 0,recivedWorks = 0;
        for(;sentWork + 1 < world_size; sentWork++)
        {
            MPI_Send(input->seq1,strlen(input->seq1) + 1, MPI_CHAR, sentWork+1, WORK, MPI_COMM_WORLD);
            MPI_Send(input->w,4, MPI_INT, sentWork+1, WORK, MPI_COMM_WORLD);
            MPI_Send(input->seq2[sentWork],strlen(input->seq2[sentWork]) + 1, MPI_CHAR, sentWork+1, WORK, MPI_COMM_WORLD);
            MPI_Send(&sentWork,1,MPI_INT,sentWork+1,WORK,MPI_COMM_WORLD);
        }
        output_fileStruct outputFile[input->seq2Count];
        while(recivedWorks < input->seq2Count)
        {
            MPI_Status status;
            output_fileStruct helperOutput;
            MPI_Recv(&helperOutput, sizeof(output_fileStruct), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            outputFile[helperOutput.lineNum].data = input->seq2[helperOutput.lineNum];
            outputFile[helperOutput.lineNum].mutant = helperOutput.mutant;
            recivedWorks++;
            if(sentWork < input->seq2Count)
            {
                
                MPI_Send(input->seq2[sentWork],strlen(input->seq2[sentWork]) + 1, MPI_CHAR, status.MPI_SOURCE, WORK, MPI_COMM_WORLD);
                MPI_Send(&sentWork,1,MPI_INT,status.MPI_SOURCE,WORK,MPI_COMM_WORLD);
                sentWork++;
            }
            else
            {
                MPI_Send(input->seq2[0],0, MPI_CHAR, status.MPI_SOURCE, STOP, MPI_COMM_WORLD);      // dummy message to stop the slave
            }
        }
        writeToFile(outputFile,input->seq2Count);
}

void slave(int rank)
{
    char seq2[5001];
    char seq1[5001];
    int w[4];
    MPI_Status status;
    MPI_Recv(seq1, 5001, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(w, 4, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(seq2, 5001, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    while(status.MPI_TAG != STOP)
    {
        output_fileStruct outputFile;
        MPI_Recv(&outputFile.lineNum, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);            // receive line number
        printf("recieved line: %d, seq2: %s\n",outputFile.lineNum,seq2);
        if(rank %2==0)          // if the rank is even - > send to cuda   
        {
            int* result = getBestMutantCuda(seq1,seq2,w);
            outputFile.mutant.bestOffset = result[0];
            outputFile.mutant.n = result[1];
            outputFile.mutant.k = result[2];
            free(result);
        }
        else
        {
            outputFile.mutant = getBestMutant(seq1, seq2, w);
        }

        MPI_Send(&outputFile, sizeof(output_fileStruct), MPI_BYTE, 0, 0, MPI_COMM_WORLD);                       // send output
        MPI_Recv(seq2, 5001, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char *argv[])
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Input* input;
    if (rank == 0)
    {
        input = readFromFile();
        master(input, world_size);
        freeFunc(input);
    }
    else
    {
        slave(rank);
    }
    MPI_Finalize();
}
