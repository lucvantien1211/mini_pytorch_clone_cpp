#pragma once

#include <memory>
#include <string>

namespace mt {

class Node {
   public:
    virtual ~Node() = default;

    virtual std::string name() const = 0;
};

}  // namespace mt