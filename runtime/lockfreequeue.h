#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include <coroutine>
#include <vector>

using namespace std;
template <typename T> class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity) : capacity_(capacity + 1), buffer_(capacity + 1), head_(0), tail_(0) {}

    bool push(const T &item) {
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

    bool pop(T &item) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            // Queue is empty
            return false;
        }

        item = buffer_[tail];
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return true;
    }
    T &operator[](int i) { return buffer_[i]; }
    const size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};
struct SharedData {
    int i;
    int *a;
    int *b;
    int res;
    bool valid;
};

struct Task {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type {
        auto get_return_object() { return Task{handle_type::from_promise(*this)}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };

    handle_type handle;

    Task(handle_type h) : handle(h) {}
    ~Task() {
        if (handle)
            handle.destroy();
    }
    Task(const Task &) = delete;
    Task &operator=(const Task &) = delete;
    Task(Task &&other) : handle(other.handle) { other.handle = nullptr; }
    Task &operator=(Task &&other) {
        if (this != &other) {
            if (handle)
                handle.destroy();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    bool done() const { return handle.done(); }
    void resume() { handle.resume(); }
};
struct remote_result {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct promise_type {
        int value;
        auto get_return_object() { return remote_result{handle_type::from_promise(*this)}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_value(int v) { value = v; }
        void unhandled_exception() {}
    };

    handle_type handle;

    remote_result(handle_type h) : handle(h) {}
    ~remote_result() {
        if (handle)
            handle.destroy();
    }
    remote_result(const remote_result &) = delete;
    remote_result &operator=(const remote_result &) = delete;
    remote_result(remote_result &&other) : handle(other.handle) { other.handle = nullptr; }
    remote_result &operator=(remote_result &&other) {
        if (this != &other) {
            if (handle)
                handle.destroy();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    bool done() const { return handle.done(); }
    void resume() { handle.resume(); }
    int get_result() const { return handle.promise().value; }
};

#endif // LOCK_FREE_QUEUE_H