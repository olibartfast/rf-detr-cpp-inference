// Regenerates the golden CPU fixtures under tests/data/gpu_parity/.
//
// Run once (from the repo root) and commit the output; the tests read these
// files and never regenerate them. See tests/data/gpu_parity/README.md for the
// exact geometry, resolution, and seeds.
//
//   ./build/gpu_parity_gen tests/data/gpu_parity

#include "gpu_parity_fixtures.hpp"

#include <iostream>
#include <memory>

namespace {

class TempLabelFile {
  public:
    explicit TempLabelFile(size_t count) : path_(std::filesystem::temp_directory_path() / "gpu_parity_gen_labels.txt") {
        std::ofstream file(path_);
        for (size_t i = 0; i < count; ++i) {
            file << "class" << i << '\n';
        }
    }
    ~TempLabelFile() { std::filesystem::remove(path_); }
    [[nodiscard]] const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

void generate_natural(const gpu_parity::NaturalSpec &spec, uint32_t image_seed, uint32_t output_seed,
                      const std::filesystem::path &dir, const std::filesystem::path &labels) {
    const auto image = gpu_parity::make_test_image(spec.width, spec.height, image_seed);
    // JPEG, not PNG: the DALI EncodedImage pipeline decodes with nvJPEG, which
    // only reads JPEG. The stored preprocessed tensor is derived from the
    // decoded bytes (what a reader gets back), not the pre-encode pixels.
    const auto image_path = dir / (std::string(spec.name) + ".jpg");
    if (!rfdetr::media::save_image(image, image_path)) {
        throw std::runtime_error("failed to write image: " + image_path.string());
    }

    const auto decoded = rfdetr::media::load_image(image_path);
    if (decoded.empty()) {
        throw std::runtime_error("failed to re-read image: " + image_path.string());
    }
    const auto tensor = gpu_parity::cpu_preprocess(decoded);
    gpu_parity::write_floats(dir / (std::string(spec.name) + ".preprocessed.bin"), tensor);

    const auto outputs = gpu_parity::make_synthetic_outputs(gpu_parity::kNaturalQueries, gpu_parity::kNaturalClasses,
                                                            gpu_parity::kNaturalMaskSize, gpu_parity::kNaturalMaskSize,
                                                            gpu_parity::kNaturalHits, output_seed);
    gpu_parity::write_outputs(dir / (std::string(spec.name) + ".outputs.bin"), outputs);

    gpu_parity::Expected expected{spec.width, spec.height,
                                  gpu_parity::cpu_decode(outputs, labels, spec.width, spec.height)};
    gpu_parity::write_expected(dir / (std::string(spec.name) + ".expected.txt"), expected);

    std::cout << "generated " << spec.name << ": " << expected.detections.size() << " detections, " << tensor.size()
              << " preprocessed floats\n";
}

void generate_dense(const std::filesystem::path &dir, const std::filesystem::path &labels) {
    const auto outputs = gpu_parity::make_synthetic_outputs(gpu_parity::kDenseQueries, gpu_parity::kDenseClasses,
                                                            gpu_parity::kDenseMaskSize, gpu_parity::kDenseMaskSize,
                                                            gpu_parity::kDenseHits, gpu_parity::kDenseSeed);
    gpu_parity::write_outputs(dir / "dense.outputs.bin", outputs);

    gpu_parity::Expected expected{
        gpu_parity::kDenseOrigW, gpu_parity::kDenseOrigH,
        gpu_parity::cpu_decode(outputs, labels, gpu_parity::kDenseOrigW, gpu_parity::kDenseOrigH)};
    gpu_parity::write_expected(dir / "dense.expected.txt", expected);

    std::cout << "generated dense: " << expected.detections.size() << " detections\n";
}

} // namespace

int main(int argc, const char *argv[]) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : "tests/data/gpu_parity";
    std::filesystem::create_directories(dir);

    TempLabelFile labels(9);

    uint32_t index = 0;
    for (const auto &spec : gpu_parity::kNaturalFixtures) {
        generate_natural(spec, 10 + index, 100 + index, dir, labels.path());
        ++index;
    }
    generate_dense(dir, labels.path());
    return 0;
}
