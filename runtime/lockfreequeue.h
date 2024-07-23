#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include <vector>
using namespace std;
template<typename T>
class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity)
        : capacity_(capacity), buffer_(capacity), head_(0), tail_(0) {}

    bool push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) % capacity_;

        if (next_head == tail_.load(std::memory_order_acquire)) {
            // Queue is full
            return false;
        }

        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            // Queue is empty
            return false;
        }

        item = buffer_[tail];
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return true;
    }
    T &operator[](int i){
        return buffer_[i];
    }

private:
    const size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};
struct SharedData {
    int i;
    int* a;
    int* b;
    int res;
    bool valid;
};
#endif // LOCK_FREE_QUEUE_H