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

remote_result remote_async(int a, int b[], int c[]) {
    atomicQueue.push({a, b, c, 0, true});
    if (atomicQueue[a].res == 0) {
        co_await std::suspend_always{};
    }
    co_return atomicQueue[a].res;
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
    std::vector<remote_result> futures;
    for (int i = 0; i < M - 1; i += 4) {
        futures.push_back(remote_async(i/4, a, b));
    }
       for (auto& result : futures) {
        while (!result.handle.done()) {
            result.handle.resume();
            std::this_thread::yield();
        }
        c += result.get_result();
    }
    printf("c=%d\n", c);
    return 0;
}