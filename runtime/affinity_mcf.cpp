#include "utils.h"
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <unistd.h>
#include <x86intrin.h>

Channel<SharedData, 16> data_to;
Channel<ResultData, 16> data_back;
// int remote(int a, int b[], int c[]) { // how to make it async?
//     atomicQueue.push({a, b, c, 0, true});
//     while (atomicQueue[a].res == 0) {
//         usleep(1);
//     }
//     return atomicQueue[a].res;
// };

// remote_result remote_async(int a, int b[], int c[]) {
//     atomicQueue.push({a, b, c, 0, true});
//     while (true) {
//         if (atomicQueue[a].res == 0)
//             co_await std::suspend_always{};

//         if (atomicQueue[a].res != 0) {
//             co_return atomicQueue[a].res;
//         }
//     }
// }
#define N 100
#define M 100
#define K 4
remote_result remote_async(int a, int b[], int c[]) {
    while (true) {
        SharedData data = {a * K, b, c};
        while (!data_to.send(data)) {
            co_await std::suspend_always{};
        };
        ResultData back;
        while (!data_back.receive(back)) {
            co_await std::suspend_always{};
        };
        co_return back.i;
    }
}

int a[N];
int b[N];
int local_func() {
    int c = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }
    std::vector<remote_result> futures;
    for (int i = 0; i < M; i += K) {
        futures.push_back(remote_async(i / K, a, b));
    }
    for (auto &result : futures) {
        while (!result.handle.done()) {
            result.handle.resume();
            std::this_thread::yield();
        }

        // Now it's safe to get the result
        auto d = result.get_result();
        printf("d=%d\n", d);
        c += result.get_result();
    }
    printf("c=%d\n", c);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
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
    while (true) {
        SharedData shared;
        while (!data_to.receive(shared)) {
            co_await std::suspend_always{};
        };
        ResultData return_data = {remote1(shared.i, shared.a, shared.b)};
        while (!data_back.send(return_data)) {
            co_await std::suspend_always{};
        }
    }
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
    pthread_join(local_thread, nullptr);
    pthread_join(remote_thread, nullptr);

    return 0;
}