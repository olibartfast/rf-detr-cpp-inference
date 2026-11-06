#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rfdetr::backend::litert_detail {
size_t element_count(const std::vector<int64_t> &shape);
void validate_input(const std::vector<int64_t> &shape, const std::vector<int64_t> &requested);
void validate_outputs(const std::vector<std::vector<int64_t>> &shapes);
} // namespace rfdetr::backend::litert_detail
