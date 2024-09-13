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

Channel<SharedDataMCF, 16> data_to;
Channel<ResultDataMCF, 16> data_back;

std::vector<remote_result> futures;
int counter = 0;

#define M 204416
#define K 1
remote_result remote_async(arc_t *arc, long *basket_size, BASKET **perm) {
    while (true) {
        SharedDataMCF data = {arc, basket_size, perm};
        while (!data_to.send(data)) {
            co_await std::suspend_always{};
        };
        ResultDataMCF back;
        while (!data_back.receive(back)) {
            co_await std::suspend_always{};
        };
        co_return back.i;
    }
}
extern "C" {
void remote(arc_t *arc, long *basket_size, BASKET *perm[]) {
    counter++;
    // printf("counter: %d\n", counter);
    // usleep(100000);
    futures.push_back(remote_async(arc, basket_size, perm));
    if (counter % K == 0) {
        for (auto &result : futures) {
            while (!result.handle.done()) {
                result.handle.resume();
                std::this_thread::yield();
            }
        }
        futures.clear();
        //   printf("counter: %d\n", counter);

    }
}
}

// void (*remote1)(arc_t *arc, long *basket_size, BASKET *perm[]);
int bea_is_dual_infeasible(arc_t *arc, cost_t red_cost) {
    return ((red_cost < 0 && arc->ident == 1) || (red_cost > 0 && arc->ident == 2));
};
void remote1(arc_t *arc, long *basket_size, BASKET *perm[]) {
    /* red_cost = bea_compute_red_cost( arc ); */
    cost_t red_cost;
    // printf("arc: %p %p\n", arc, basket_size);
    if (arc->ident > 0) {
        /* red_cost = bea_compute_red_cost( arc ); */
        red_cost = arc->cost - arc->tail->potential + arc->head->potential;
        if (bea_is_dual_infeasible(arc, red_cost)) {
            (*basket_size)++;
            perm[*basket_size]->a = arc;
            perm[*basket_size]->cost = red_cost;
            perm[*basket_size]->abs_cost = ABS(red_cost);
        }
    }
}
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
    auto start = std::chrono::high_resolution_clock::now();
    main_ptr();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
Task process_queue_item(size_t i) {
    while (true) {
        SharedDataMCF shared;
        while (!data_to.receive(shared)) {
            co_await std::suspend_always{};
        };
        remote1(shared.arc, shared.basket_size, shared.perm);
        ResultDataMCF return_data = {(long)i};
        while (!data_back.send(return_data)) {
            co_await std::suspend_always{};
        }
    }
}
// Remote thread function
void *remote_thread_func(void *arg) {
    set_cpu_affinity(64);
    // void *handle = dlopen("./libremote.so", RTLD_NOW | RTLD_GLOBAL);
    // printf("handle: %p %s\n", handle,dlerror());
    // if (!handle) {
    //     exit(-1);
    // }
    // remote1 = (void(*)(arc_t * arc, long *basket_size, basket *perm[])) dlsym(handle, "remote1");
    std::vector<Task> futures;
    for (size_t i = 0; i < M; i += 1) {
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