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

Channel<SharedDataBTree, 16> data_to;
Channel<bool, 16> data_back;

Task remote_async(int a, BTreeNode *b) {
    while (true) {
        SharedDataBTree data = {a, b};
        while (!data_to.send(data)) {
            co_await std::suspend_always{};
        };
        bool data_;
        while (!data_back.receive(data_)) {
            co_return;
        };
        co_return;
    }
}
#define N 100000000
#define M 100000000
#define K 100

int local_func() {
    auto start = std::chrono::high_resolution_clock::now();
    BTreeNode *root = create_node(true);
    std::vector<Task> futures;
    for (int i = 0; i < M; i += K) {
        futures.push_back(remote_async(i, root));
    }
    for (auto &result : futures) {
        while (!result.handle.done()) {
            result.handle.resume();
            std::this_thread::yield();
        }

        // Now it's safe to get the result
        // auto d = result.get_result();
        // printf("d=%d\n", d);
    }
    // print_btree(root, 0);
    // usleep(1000);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
void (*remote1)(BTreeNode **root, int key);
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
Task process_queue_item() {
    while (true) {
        SharedDataBTree shared;
        while (!data_to.receive(shared)) {
            co_await std::suspend_always{};
        };
        remote1(&shared.a, shared.i);
        while (!data_back.send(true)) {
            co_return;
        };
        co_return;
    }
}
// Remote thread function
void *remote_thread_func(void *arg) {
    set_cpu_affinity(64);
    void *handle = dlopen("./libremote.so", RTLD_NOW | RTLD_GLOBAL);
    printf("handle: %p\n", handle);
    if (!handle) {
        exit(-1);
    }
    // printf("error: %s",dlerror());
    // dlerror();
    remote1 = (void (*)(BTreeNode **, int))dlsym(handle, "insert");
    std::vector<Task> futures;
    for (size_t i = 0; i < M / K; i += 1) {
        futures.push_back(process_queue_item());
    }
    for (auto &result : futures) {
        while (!result.handle.done()) {
            result.handle.resume();
            std::this_thread::yield();
        }
    }
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