#include "lockfreequeue.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
extern LockFreeQueue<SharedData> atomicQueue;
int remote(int a, int b[], int c[]) {
    atomicQueue.push({a, b, c, 0, true});
    while (atomicQueue[a].res == 0) {
        usleep(1);
    }
    return atomicQueue[a].res;
};
#define N 100000000
#define M 10000000
int a[N];
int b[N];
int local_func() {
    int c = 0;
    for (int i = 0; i < N; i++) {
        a[i] = rand() % N;
        b[i] = rand() % N;
    }
    for (int i = 0; i < M - 1; i += 4) {
        c += remote(i/4, a, b);
    }
    printf("c=%d\n", c);
    return 0;
}