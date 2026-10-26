#pragma once

#include <memory>
#include <string>
#include <thread>

namespace mt {
class Tensor;

class Node {
   public:
    virtual ~Node() = default;

    virtual std::string name() const = 0;

    virtual void backward(const Tensor& grad_output) = 0;
};

inline bool& grad_enabled_state() {
    static thread_local bool enabled = true;
    return enabled;
}

inline bool is_grad_enabled() { return grad_enabled_state(); }

inline void set_grad_enabled(bool enabled) { grad_enabled_state() = enabled; }

class NoGradGuard {
   private:
    bool prev_enabled_;

   public:
    NoGradGuard() : prev_enabled_(is_grad_enabled()) { set_grad_enabled(false); }

    ~NoGradGuard() { set_grad_enabled(prev_enabled_); }

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

    NoGradGuard(NoGradGuard&&) = delete;
    NoGradGuard& operator=(NoGradGuard&&) = delete;
};

}  // namespace mt