#include "dali_preprocessor.hpp"

#ifdef USE_DALI

#include "cuda_check.hpp"

#include <cstdlib>
#include <dali/c_api.h>
#include <dali/operators.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rfdetr::gpu {

namespace {

/// Process-global initialisation, exactly once before any pipeline is created.
///
/// Both calls are required. daliInitialize() brings up the DALI backend;
/// daliInitOperators() registers the operator schemas from libdali_operators.so,
/// which nothing loads implicitly — DALI's Python bindings dlopen it, and a C++
/// host must ask for it. Without the second call, deserialisation succeeds and
/// the first run fails with `No schema found for operator "decoders__Image"`.
void ensure_dali_initialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        daliInitialize();
        daliInitOperators();
    });
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open serialized DALI pipeline: " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

const char *dtype_name(dali_data_type_t type) {
    switch (type) {
    case DALI_UINT8:
        return "UINT8";
    case DALI_FLOAT:
        return "FLOAT";
    case DALI_FLOAT16:
        return "FLOAT16";
    default:
        return "other";
    }
}

} // namespace

struct DaliPreprocessor::Impl {
    daliPipelineHandle handle{};
    Source source;
    std::vector<std::int64_t> last_shape;
    bool output_shared{false};

    Impl(const std::filesystem::path &path, Source src, int device_id) : source(src) {
        ensure_dali_initialized();

        const std::string serialized = read_file(path);
        if (serialized.empty()) {
            throw std::runtime_error("Serialized DALI pipeline is empty: " + path.string());
        }
        if (daliIsDeserializable(serialized.c_str(), static_cast<int>(serialized.size())) != 0) {
            throw std::runtime_error("File is not a serialized DALI pipeline: " + path.string() +
                                     " (regenerate it with deploy/dali/generate_preprocess_pipeline.py)");
        }

        daliDeserializeDefault(&handle, serialized.c_str(), static_cast<int>(serialized.size()));

        // Fail loudly at construction if the pipeline does not match the variant
        // the caller asked for, rather than producing wrong tensors later.
        const char *expected = src == Source::EncodedImage ? kDaliEncodedInputName : kDaliFrameInputName;
        const int num_inputs = daliGetNumExternalInput(&handle);
        bool found = false;
        std::string names;
        for (int i = 0; i < num_inputs; ++i) {
            const char *name = daliGetExternalInputName(&handle, i);
            if (name == nullptr) {
                continue;
            }
            if (!names.empty()) {
                names += ", ";
            }
            names += name;
            if (std::string(name) == expected) {
                found = true;
            }
        }
        if (!found) {
            daliDeletePipeline(&handle);
            throw std::runtime_error("Serialized DALI pipeline " + path.string() + " has no external source named '" +
                                     expected + "' (found: " + (names.empty() ? "none" : names) + ")");
        }

        const unsigned num_outputs = daliGetNumOutput(&handle);
        if (num_outputs < 1) {
            daliDeletePipeline(&handle);
            throw std::runtime_error("Serialized DALI pipeline has no outputs: " + path.string());
        }
        if (daliGetOutputDevice(&handle, 0) != device_type_t::GPU) {
            daliDeletePipeline(&handle);
            throw std::runtime_error("DALI pipeline output 0 must live on the GPU: " + path.string());
        }
        (void)device_id;
    }

    ~Impl() {
        if (output_shared) {
            daliOutputRelease(&handle);
        }
        daliDeletePipeline(&handle);
    }

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;
};

DaliPreprocessor::DaliPreprocessor(const std::filesystem::path &serialized_pipeline, Source source, int device_id)
    : impl_(std::make_unique<Impl>(serialized_pipeline, source, device_id)) {}

DaliPreprocessor::~DaliPreprocessor() = default;

const std::vector<std::int64_t> &DaliPreprocessor::last_output_shape() const noexcept { return impl_->last_shape; }

DaliPreprocessor::Source DaliPreprocessor::source() const noexcept { return impl_->source; }

void DaliPreprocessor::process_encoded(std::span<const std::uint8_t> bytes, void *dst_device, std::size_t dst_bytes,
                                       StreamHandle stream) {
    if (impl_->source != Source::EncodedImage) {
        throw std::runtime_error("process_encoded called on a BgrFrame DALI pipeline");
    }
    if (bytes.empty()) {
        throw std::runtime_error("process_encoded received no data");
    }

    const std::int64_t shape[1] = {static_cast<std::int64_t>(bytes.size())};
    // DALI_ext_force_copy: `bytes` is caller-owned and may be reused or freed as
    // soon as this returns, so the pipeline must not alias it.
    daliSetExternalInputAsync(&impl_->handle, kDaliEncodedInputName, device_type_t::CPU, bytes.data(), DALI_UINT8,
                              shape, 1, nullptr, static_cast<cudaStream_t>(stream), DALI_ext_force_copy);

    run_and_copy(dst_device, dst_bytes, stream);
}

void DaliPreprocessor::process_frame(const void *bgr_device, int height, int width, void *dst_device,
                                     std::size_t dst_bytes, StreamHandle stream) {
    if (impl_->source != Source::BgrFrame) {
        throw std::runtime_error("process_frame called on an EncodedImage DALI pipeline");
    }
    if (bgr_device == nullptr || height <= 0 || width <= 0) {
        throw std::runtime_error("process_frame received an invalid frame");
    }

    const std::int64_t shape[3] = {height, width, 3};
    daliSetExternalInputAsync(&impl_->handle, kDaliFrameInputName, device_type_t::GPU, bgr_device, DALI_UINT8, shape, 3,
                              "HWC", static_cast<cudaStream_t>(stream), DALI_ext_force_copy);

    run_and_copy(dst_device, dst_bytes, stream);
}

void DaliPreprocessor::run_and_copy(void *dst_device, std::size_t dst_bytes, StreamHandle stream) {
    if (dst_device == nullptr) {
        throw std::runtime_error("DALI preprocessing destination is null");
    }

    daliRun(&impl_->handle);
    daliShareOutput(&impl_->handle);
    impl_->output_shared = true;

    // Validate the tensor before copying: a pipeline generated for a different
    // resolution would otherwise silently overflow or underfill the binding.
    const dali_data_type_t type = daliTypeAt(&impl_->handle, 0);
    if (type != DALI_FLOAT) {
        daliOutputRelease(&impl_->handle);
        impl_->output_shared = false;
        throw std::runtime_error(std::string("DALI pipeline output 0 must be FLOAT, got ") + dtype_name(type));
    }

    // daliMaxDimTensors reports the *actual* rank. daliGetDeclaredOutputNdim
    // reports the optionally-declared one, which these pipelines do not set, so
    // it comes back as a sentinel that turns into a nonsense size.
    const std::size_t ndim = daliMaxDimTensors(&impl_->handle, 0);
    // daliShapeAtSample hands over malloc'd memory that the caller must free.
    std::int64_t *shape = daliShapeAtSample(&impl_->handle, 0, 0);
    if (shape != nullptr) {
        impl_->last_shape.assign(shape, shape + ndim);
        std::free(shape);
    } else {
        impl_->last_shape.clear();
    }

    const std::size_t produced = daliTensorSize(&impl_->handle, 0);
    if (produced != dst_bytes) {
        std::ostringstream message;
        message << "DALI pipeline produced " << produced << " bytes but the destination expects " << dst_bytes
                << " (shape";
        for (const auto dim : impl_->last_shape) {
            message << ' ' << dim;
        }
        message << "). Regenerate the pipeline for this model's input resolution.";
        daliOutputRelease(&impl_->handle);
        impl_->output_shared = false;
        throw std::runtime_error(message.str());
    }

    daliOutputCopy(&impl_->handle, dst_device, 0, device_type_t::GPU, static_cast<cudaStream_t>(stream),
                   DALI_use_copy_kernel);

    // Synchronise before releasing. daliOutputRelease returns the tensor to
    // DALI's buffer pool, and the copy above is asynchronous on `stream`:
    // releasing first would let a subsequent iteration overwrite the source
    // while the copy is still in flight, which corrupts data intermittently
    // rather than failing. The wait costs one device-to-device copy of the
    // input tensor (~0.1 ms at 560x560), not a full pipeline stall.
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));

    daliOutputRelease(&impl_->handle);
    impl_->output_shared = false;
}

} // namespace rfdetr::gpu

#endif // USE_DALI
