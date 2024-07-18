#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t NUM_ACCESSES = 1000000;
constexpr size_t NUM_THREADS = 4;  // Number of threads

// Function to align memory to cache line boundary
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}

// Function to free aligned memory
void aligned_free(void* ptr) {
    free(ptr);
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

// Structure to pass arguments to threads
struct ThreadArgs {
    int* buffer;
    size_t num_accesses;
    int cpu;
};

// Thread function to perform ping-pong cacheline test
void* remote_thread(void* args) {
    ThreadArgs* threadArgs = static_cast<ThreadArgs*>(args);
    int* buffer = threadArgs->buffer;
    size_t num_accesses = threadArgs->num_accesses;
    int cpu = threadArgs->cpu;

    // Set CPU affinity for this thread
    set_cpu_affinity(cpu);

    // dlopen

    return nullptr;
}

int main() {
    size_t num_elements = NUM_ACCESSES * 2;
    size_t buffer_size = num_elements * sizeof(int);

    // Allocate memory aligned to cache line size
    int* buffer = static_cast<int*>(aligned_malloc(buffer_size, CACHE_LINE_SIZE));
    if (!buffer) {
        std::cerr << "Memory allocation failed" << std::endl;
        return 1;
    }

    // dlopen
    // Initialize buffer to ensure it is paged in
    std::memset(buffer, 0, buffer_size);

    pthread_t threads[NUM_THREADS];
    ThreadArgs threadArgs[NUM_THREADS];

    auto start = std::chrono::high_resolution_clock::now();

    // Create threads to perform ping-pong cacheline test
    for (int i = 0; i < NUM_THREADS; ++i) {
        threadArgs[i] = { buffer, NUM_ACCESSES / NUM_THREADS, i };
        if (pthread_create(&threads[i], nullptr, remote_thread, &threadArgs[i]) != 0) {
            std::cerr << "Error creating thread" << std::endl;
            aligned_free(buffer);
            return 1;
        }
    }

    // Join threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        if (pthread_join(threads[i], nullptr) != 0) {
            std::cerr << "Error joining thread" << std::endl;
            aligned_free(buffer);
            return 1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Ping-pong cacheline test duration: " << duration.count() << " seconds" << std::endl;

    aligned_free(buffer);
    return 0;
}