#include "../test/mcf_remote.h"
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

Channel<SharedDataMCF, 16> *data_to;
Channel<ResultDataMCF, 16> *data_back;
int (*main_ptr)(int, char **);
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

cost_t (*remote1)(arc_t *arc, long *basket_size, BASKET *perm[]);
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
int local_func() {
    set_cpu_affinity(0);
    printf("Address of main: %p\n", main_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    main_ptr(0, NULL);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
Task process_queue_item(size_t i) {
    while (true) {
        SharedDataMCF shared;
        while (!data_to->receive(shared)) {
            co_await std::suspend_always{};
        };
        ResultDataMCF return_data = {remote1(shared.arc, shared.basket_size, shared.perm)};
        while (!data_back->send(return_data)) {
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
    remote1 = (cost_t(*)(arc_t * arc, long *basket_size, basket *perm[])) dlsym(handle, "remote1");
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
    void *handle = dlopen("/home/yangyw/isca25/CIRA/bench/mcf/libmcf.so",
                          RTLD_NOW | RTLD_GLOBAL); // Get handle to the current process
    if (handle == NULL) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    main_ptr = (int (*)(int, char **))dlsym(handle, "main_ptr");
    if (main_ptr == NULL) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    data_to = (Channel<SharedDataMCF, 16> *)dlsym(handle, "data_to");
    data_back = (Channel<ResultDataMCF, 16> *)dlsym(handle, "data_back");
    pthread_t remote_thread, local_thread;
    pthread_create(&remote_thread, nullptr, remote_thread_func, nullptr);
    pthread_create(&local_thread, nullptr, local_thread_func, nullptr);

    // Wait for threads to complete
    pthread_join(local_thread, nullptr);
    pthread_join(remote_thread, nullptr);

    return 0;
}