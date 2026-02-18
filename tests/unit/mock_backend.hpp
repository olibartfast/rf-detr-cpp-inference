#pragma once

#include "backends/inference_backend.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

/// A mock inference backend for unit testing.
/// Pre-configure output tensors and shapes before calling run_inference.
class MockBackend : public rfdetr::backend::InferenceBackend {
  public:
    /// Set the output tensors that will be returned after run_inference.
    void set_outputs(std::vector<std::vector<float>> data, std::vector<std::vector<int64_t>> shapes) {
        output_data_ = std::move(data);
        output_shapes_ = std::move(shapes);
    }

    std::vector<int64_t> initialize(const std::filesystem::path & /*model_path*/,
                                    const std::vector<int64_t> &input_shape) override {
        return input_shape;
    }

    std::vector<void *> run_inference(std::span<const float> /*input_data*/,
                                      const std::vector<int64_t> & /*input_shape*/) override {
        return {};
    }

    [[nodiscard]] size_t get_output_count() const override { return output_data_.size(); }

    void get_output_data(size_t output_index, float *data, size_t size) override {
        if (output_index >= output_data_.size()) {
            throw std::out_of_range("Output index out of range");
        }
        const auto copy_size = std::min(size, output_data_[output_index].size());
        std::memcpy(data, output_data_[output_index].data(), copy_size * sizeof(float));
    }

    [[nodiscard]] std::vector<int64_t> get_output_shape(size_t output_index) const override {
        if (output_index >= output_shapes_.size()) {
            throw std::out_of_range("Shape index out of range");
        }
        return output_shapes_[output_index];
    }

    [[nodiscard]] std::string get_backend_name() const override { return "MockBackend"; }

  private:
    std::vector<std::vector<float>> output_data_;
    std::vector<std::vector<int64_t>> output_shapes_;
};
