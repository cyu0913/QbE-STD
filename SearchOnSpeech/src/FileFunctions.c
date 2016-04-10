/*********************************************************************
 *  Copyright (C) Paula López Otero, GTM, Universidade de Vigo,2015  *
 *  E-mail: plopez@gts.uvigo.es                                      *
 * *******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <armadillo>
using namespace arma;

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS

//Struct for storing the experiments required for the phoneme unit selection approach
typedef struct {

  char *match;
  char *query;
  int start;
  int duration;
  Mat<float> matchData;
  Mat<float> queryData;
} Experiment;

union byte2 {
  char byte[2];
  short int numint;
};

union byte4 {
  char byte[4];
  int numint;
  float numfloat;
};

//-------- New -------------

void endianSwap4(union byte4 *un) {
    // swap
    char c1 = (*un).byte[0];
    (*un).byte[0] = (*un).byte[3];
    (*un).byte[3] = c1;
    c1 = (*un).byte[1];
    (*un).byte[1] = (*un).byte[2];
    (*un).byte[2] = c1;
}
//----------------------------

short int endianSwap2int(short int a) {
  union byte2 un;
  un.numint = a;

  // swap
  char c1 = un.byte[0];
  un.byte[0] = un.byte[1];
  un.byte[1] = c1;

  return un.numint;
}

int endianSwap4int(int a) {
  union byte4 un;
  un.numint = a;

  // swap
  char c1 = un.byte[0];
  un.byte[0] = un.byte[3];
  un.byte[3] = c1;
  c1 = un.byte[1];
  un.byte[1] = un.byte[2];
  un.byte[2] = c1;

  return un.numint;
}

float endianSwap4float(float a) {
  union byte4 un;
  un.numfloat = a;

  // swap
  char c1 = un.byte[0];
  un.byte[0] = un.byte[3];
  un.byte[3] = c1;
  c1 = un.byte[1];
  un.byte[1] = un.byte[2];
  un.byte[2] = c1;

  return un.numfloat;
}

//Function for reading queries and documents stored in HTK format
//Modified to read big edian machine code, in cluster
Mat<float> readFile(char *dataFile) {

  FILE *f = fopen(dataFile,"rb");

  int headerInt[2];
  short headerShort[2];

  fread(headerInt,sizeof(int),2,f);
  fread(headerShort,sizeof(short),2,f);

  int nSamples = endianSwap4int(headerInt[0]);
  int nFeatures = endianSwap2int(headerShort[0])/sizeof(float);


  Mat<float> data = Mat<float>(nFeatures,nSamples);
  int i,j;

  float featureVector[nFeatures];
  for(i = 0; i < nSamples; i++) {
    fread(featureVector,4,nFeatures,f);
    for(j = 0; j < nFeatures; j++) {
      data(j,i) = endianSwap4float(featureVector[j]);
    }
  }
  fclose(f);
  return(data);

}

// Orginal Read File Function, works on litte edian
Mat<float> readFileOrg(char *dataFile) {

  FILE *f = fopen(dataFile,"rb");

  int headerInt[2];
  short headerShort[2];

  fread(headerInt,sizeof(int),2,f);
  fread(headerShort,sizeof(short),2,f);

  int nSamples = headerInt[0];
  int nFeatures = headerShort[0]/sizeof(float);


  Mat<float> data = Mat<float>(nFeatures,nSamples);
  int i,j;

  float featureVector[nFeatures];
  for(i = 0; i < nSamples; i++) {
    fread(featureVector,4,nFeatures,f);
    for(j = 0; j < nFeatures; j++) {
      data(j,i) = featureVector[j];
    }
  }
  fclose(f);
  return(data);

}

//Function for reading the list of relevant phoneme units
Col<int> readKeepFile(char *dataFile) {

  FILE *f = fopen(dataFile,"r");

  int i, n = 0;

  while(1) {
    fscanf(f,"%*d");
    if(feof(f)) {
      break;
    }
    n++;
  }
  fclose(f);

  Col<int> data(n);

  f = fopen(dataFile,"r");
  int feature;
  for(i = 0; i < n; i++) {
    fscanf(f,"%d",&feature);
    data[i] = feature;
  }
  fclose(f);
  return(data);

}

//Function for reading the sample period from an HTK file
int getSamplePeriod(char *dataFile) {

  FILE *f = fopen(dataFile,"r");
  int headerInt[2];
  fread(headerInt,sizeof(int),2,f);
  fclose(f);
  return(endianSwap4int(headerInt[1]));

}

//Function for reading a time segment of an HTK file (in order to handle the case of spoken term detection,
//where we know the start and end times of the match between the query and the document).
Mat<float> readFileSegment(char *dataFile,int start,int duration) {

  FILE *f = fopen(dataFile,"rb");

  int headerInt[2];
  short headerShort[2];

  fread(headerInt,sizeof(int),2,f);
  fread(headerShort,sizeof(short),2,f);

  int nSamples = duration;
  int nFeatures = headerShort[0]/sizeof(float);
  Mat<float> data = Mat<float>(nFeatures,nSamples);
  int i,j;

  float featureVector[nFeatures];

  for(i = 0; i < start; i++) {
    fread(featureVector,4,nFeatures,f);
  }

  for(i = 0; i < duration; i++) {
    fread(featureVector,4,nFeatures,f);
    for(j = 0; j < nFeatures; j++) {
      data(j,i) = featureVector[j];
    }
  }
  fclose(f);
  return(data);

}

//Computation of the number of experiments to be run for phoneme unit selection
int computeNumberOfExperiments(char *experimentsFile) {
  
  FILE *f = fopen(experimentsFile,"r");
  
  int counter = 0;
  
  while(1) {
    fscanf(f,"%*s %*s %*f %*f");
    if(feof(f)) {
      break;
    }
    counter++;
  }
  fclose(f);
  return(counter);
}

//Function for reading the experiments file, in format "MatchFile QueryFile MatchStart MatchDuration"
void readExperimentsFile(char *experimentsFile,Experiment* exps,int nExperiments) {
  
  FILE *f = fopen(experimentsFile,"r");
  
  int i;

  char match[200],query[200];
  float start,duration;
  for(i = 0; i < nExperiments; i++) {
    fscanf(f,"%s %s %f %f\n",match,query,&start,&duration);
    exps[i].match = match;
    exps[i].query = query;
    exps[i].start = (int)(start*100);
    exps[i].duration = (int)(duration*100);
    exps[i].matchData = readFileSegment(exps[i].match,exps[i].start,exps[i].duration);
    exps[i].queryData = readFile(exps[i].query);
  }
  fclose(f);
}
