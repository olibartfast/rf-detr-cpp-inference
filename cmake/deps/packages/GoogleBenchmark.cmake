# Version comes from versions.env via cmake/versions.cmake.
deps_declare(GoogleBenchmark
    REQUIRED              OFF
    APT                   OFF
    CONAN_RECIPE          benchmark
    CONAN_FIND            benchmark
    CONAN_VERSION         "${GOOGLE_BENCHMARK_VERSION}"
    CONAN_TARGETS         "benchmark::benchmark"
    VCPKG_FIND            benchmark
    VCPKG_TARGETS         "benchmark::benchmark"
    PROVIDED_ACQUIRE      FETCHCONTENT
    PROVIDED_FC_REPO      "https://github.com/google/benchmark.git"
    PROVIDED_FC_TAG       "v${GOOGLE_BENCHMARK_VERSION}"
    PROVIDED_FC_TARGETS   "benchmark::benchmark"
)
