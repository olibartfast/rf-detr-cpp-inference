#pragma once

#ifdef USE_EXECUTORCH

#include "inference_backend.hpp"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <memory>

namespace rfdetr::backend {

/**
 * @brief ExecuTorch implementation of InferenceBackend
 *
 * Runs `.pte` programs produced by `rfdetr >= 1.9.0`
 * (`model.export(format="executorch", backend="xnnpack")`). The delegate is baked into the .pte at
 * export time; this backend only has to link the matching delegate library so it self-registers,
 * which CMake handles via EXECUTORCH_DELEGATE.
 */
class ExecuTorchBackend : public InferenceBackend {
  public:
    ExecuTorchBackend() = default;
    ~ExecuTorchBackend() override = default;
    ExecuTorchBackend(const ExecuTorchBackend &) = delete;
    ExecuTorchBackend &operator=(const ExecuTorchBackend &) = delete;
    ExecuTorchBackend(ExecuTorchBackend &&) = delete;
    ExecuTorchBackend &operator=(ExecuTorchBackend &&) = delete;

    std::vector<int64_t> initialize(const std::filesystem::path &model_path,
                                    const std::vector<int64_t> &input_shape) override;

    std::vector<void *> run_inference(std::span<const float> input_data,
                                      const std::vector<int64_t> &input_shape) override;

    [[nodiscard]] size_t get_output_count() const override;

    void get_output_data(size_t output_index, float *data, size_t size) override;

    [[nodiscard]] std::vector<int64_t> get_output_shape(size_t output_index) const override;

    [[nodiscard]] std::string get_backend_name() const override { return "ExecuTorch"; }

  private:
    /// Fetch output `output_index` from the last run, checking it exists and is a tensor.
    [[nodiscard]] executorch::aten::Tensor output_tensor(size_t output_index) const;

    /// Confirm the program returns boxes then logits, the order postprocessing assumes.
    void validate_output_order(const executorch::runtime::MethodMeta &meta) const;

    std::unique_ptr<executorch::extension::Module> module_;

    /// Taken from method_meta at load time, so get_output_count() is valid before the first run
    /// (RFDETRInference's constructor validates the output count immediately after initialize()).
    size_t output_count_ = 0;

    /// Results of the most recent forward(); mirrors OnnxRuntimeBackend's ort_output_tensors_.
    std::vector<executorch::runtime::EValue> output_values_;
};

} // namespace rfdetr::backend

#endif // USE_EXECUTORCH
