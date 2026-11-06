#pragma once

#include "inference_backend.hpp"

namespace rfdetr::backend {

// Native LiteRT CPU C API. SDK types stay behind the pimpl boundary.
class LiteRTBackend final : public InferenceBackend {
  public:
    LiteRTBackend();
    ~LiteRTBackend() override;
    std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                    const std::vector<int64_t> &input_shape) override;
    std::vector<void *> run_inference(std::span<const float> input_data,
                                      const std::vector<int64_t> &input_shape) override;
    [[nodiscard]] size_t get_output_count() const override;
    void get_output_data(size_t output_index, float *data, size_t size) override;
    [[nodiscard]] std::vector<int64_t> get_output_shape(size_t output_index) const override;
    [[nodiscard]] std::string get_backend_name() const override { return "LiteRT"; }

  private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace rfdetr::backend
