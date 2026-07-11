deps_declare(GoogleBenchmark
    REQUIRED              OFF
    APT                   OFF
    CONAN_RECIPE          benchmark
    CONAN_FIND            benchmark
    CONAN_VERSION         "1.9.1"
    CONAN_TARGETS         "benchmark::benchmark"
    VCPKG_FIND            benchmark
    VCPKG_TARGETS         "benchmark::benchmark"
    PROVIDED_ACQUIRE      FETCHCONTENT
    PROVIDED_FC_REPO      "https://github.com/google/benchmark.git"
    PROVIDED_FC_TAG       "v1.9.1"
    PROVIDED_FC_TARGETS   "benchmark::benchmark"
)
