
#ifndef __MAIN_H__
#define __MAIN_H__

#include <math.h>
#include <thread>
//#include <chrono>
#include <vector>
#include <mutex>
#include <malloc.h>
#include <Windows.h>
#include<stdio.h>
#include <string.h>
//#include <cstddef>
//#include <windows.h>

/*  To use this exported function of dll, include this header
 *  in your project.
 */

#ifdef BUILD_DLL
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT __declspec(dllimport)
#endif
typedef struct mypairs
{
    int total;
    int pindex1[10000];
    int pindex2[10000];
}mypairs, *mypairsPointer;



#ifdef __cplusplus
extern "C"
{
#endif

DLL_EXPORT mypairsPointer rightpairs(float pf[100][10000], int pfIndex[100], int length,  int num, int tvalues, int tvalues2,int core,int confinterval);

DLL_EXPORT mypairsPointer slowpairs(float pf[100][10000], int pfIndex[100], int length, int num,int tvalues, int tvalues2, int confinterval);

DLL_EXPORT void destroyp(mypairsPointer pp){
    delete pp;
    pp = NULL;
}

#ifdef __cplusplus
}
#endif

#endif // __MAIN_H__
