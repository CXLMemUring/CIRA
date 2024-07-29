#include "lockfreequeue.h"
#include <future>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
extern LockFreeQueue<SharedData> atomicQueue;
// int remote(int a, int b[], int c[]) { // how to make it async?
//     atomicQueue.push({a, b, c, 0, true});
//     while (atomicQueue[a].res == 0) {
//         usleep(1);
//     }
//     return atomicQueue[a].res;
// };
std::future<int> remote_async(int a, int b[], int c[]) {
    return std::async(std::launch::async, [a, b, c]() {
        atomicQueue.push({a, b, c, 0, true});
        while (atomicQueue[a].res == 0) {
            usleep(1);
        }
        return atomicQueue[a].res;
    });
}
#define N 100000000
#define M 100000000
int a[N];
int b[N];
int local_func() {
    int c = 0;
    for (int i = 0; i < N; i++) {
        a[i] = rand() % N;
        b[i] = rand() % N;
    }
    std::vector<std::future<int>> futures;
    for (int i = 0; i < M - 1; i += 4) {
        futures.push_back(remote_async(i / 4, a, b));
    }
for (auto& future : futures) {
    c += future.get();
}
    printf("c=%d\n", c);
    return 0;
}