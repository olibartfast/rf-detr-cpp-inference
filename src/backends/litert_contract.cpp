#include "litert_contract.hpp"

#include <limits>
#include <stdexcept>

namespace rfdetr::backend::litert_detail {

size_t element_count(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
        throw std::runtime_error("LiteRT: scalar tensors are unsupported");
    }
    size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / sizeof(float) / count) {
            throw std::runtime_error("LiteRT: tensor dimensions must be fixed, positive and fit in memory");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

void validate_input(const std::vector<int64_t> &shape, const std::vector<int64_t> &requested) {
    (void)element_count(shape);
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != shape[3]) {
        throw std::runtime_error("LiteRT: expected fixed square NCHW input [1,3,H,H]; export with format='litert'");
    }
    if (requested.size() != 4 || requested[0] != 1 || requested[1] != 3 ||
        (requested[2] != 0 && requested[2] != shape[2]) || (requested[3] != 0 && requested[3] != shape[3])) {
        throw std::runtime_error("LiteRT: requested input shape does not match the fixed model shape");
    }
}

void validate_outputs(const std::vector<std::vector<int64_t>> &shapes) {
    if (shapes.size() != 2 && shapes.size() != 3) {
        throw std::runtime_error("LiteRT: expected boxes, class logits and optional mask logits");
    }
    for (const auto &shape : shapes) {
        (void)element_count(shape);
    }
    if (shapes.at(0).size() != 3 || shapes.at(0)[0] != 1 || shapes.at(0)[2] != 4 || shapes.at(1).size() != 3 ||
        shapes.at(1)[0] != 1 || shapes.at(1)[1] != shapes.at(0)[1]) {
        throw std::runtime_error("LiteRT: expected positional boxes [1,Q,4] and class logits [1,Q,C]");
    }
    if (shapes.size() == 3 &&
        (shapes.at(2).size() != 4 || shapes.at(2)[0] != 1 || shapes.at(2)[1] != shapes.at(0)[1])) {
        throw std::runtime_error("LiteRT: expected masks [1,Q,H,W] or keypoints [1,Q,K,D]");
    }
}

} // namespace rfdetr::backend::litert_detail
