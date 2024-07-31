#include "lockfreequeue.h"
#include "test_local.c"
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <x86intrin.h>

LockFreeQueue<SharedData> atomicQueue(M / K); // local to struct
int (*remote1)(int, int[], int[]);
// Function to set CPU affinity
void set_cpu_affinity(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Error setting CPU affinity" << std::endl;
        exit(1);
    }
}
// Task process_queue_item(int i) {
//     if (!atomicQueue[i].valid) {
//         co_await std::suspend_always{};
//     }
//     atomicQueue[i].res = remote1(atomicQueue[i].i, atomicQueue[i].a, atomicQueue[i].b);
// }
Task process_queue_item(size_t i) {
    if (i < 0 || i >= atomicQueue.capacity_)
        co_return;

    if (!atomicQueue[i].valid)
        co_await std::suspend_always{};

    // Additional check after resuming
    if (!atomicQueue[i].valid)
        co_return;

    atomicQueue[i].res = remote1(atomicQueue[i].i * K, atomicQueue[i].a, atomicQueue[i].b);
}
// Remote thread function
void *remote_thread_func(void *arg) {
    set_cpu_affinity(64);
    void *handle = dlopen("./libremote.so", RTLD_NOW | RTLD_GLOBAL);
    // printf("handle: %p\n", handle);
    if (!handle) {
        exit(-1);
    }
    dlerror();
    remote1 = (int (*)(int, int[], int[]))dlsym(handle, "remote");
    std::vector<Task> futures;
    for (size_t i = 0; i < M / K; i += 1) {
        futures.push_back(process_queue_item(i));
    }
    for (auto &result : futures) {
        while (!result.handle.done()) {
            result.handle.resume();
            std::this_thread::yield();
        }
    }

    // for (int i = 0; i < M / 4; i += 1) {
    //     while (!atomicQueue[i].valid) {
    //         usleep(1); // how to make it async?
    //     }
    //     atomicQueue[i].res = (remote1(atomicQueue[i].i, atomicQueue[i].a, atomicQueue[i].b));
    // }
    return nullptr;
}

// Local thread function
void *local_thread_func(void *arg) {
    set_cpu_affinity(0);
    local_func();
    return nullptr;
}

int main() {
    // Create threads
    pthread_t remote_thread, local_thread;
    pthread_create(&remote_thread, nullptr, remote_thread_func, nullptr);
    pthread_create(&local_thread, nullptr, local_thread_func, nullptr);

    // Wait for threads to complete
    pthread_join(remote_thread, nullptr);
    pthread_join(local_thread, nullptr);

    return 0;
}