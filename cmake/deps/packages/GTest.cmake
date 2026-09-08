# Version comes from versions.env via cmake/versions.cmake.
deps_declare(GTest
    REQUIRED              TRUE
    APT                   OFF
    CONAN_RECIPE          gtest
    CONAN_FIND            GTest
    CONAN_VERSION         "${GTEST_VERSION}"
    CONAN_TARGETS         "GTest::gtest;GTest::gtest_main;GTest::gmock;GTest::gmock_main"
    VCPKG_FIND            GTest
    VCPKG_TARGETS         "GTest::gtest;GTest::gtest_main;GTest::gmock;GTest::gmock_main"
    PROVIDED_ACQUIRE      FETCHCONTENT
    PROVIDED_FC_REPO      "https://github.com/google/googletest.git"
    PROVIDED_FC_TAG       "release-${GTEST_VERSION}"
    PROVIDED_FC_TARGETS   "gtest;gtest_main;gmock;gmock_main"
)
