deps_declare(GTest
    REQUIRED              TRUE
    APT                   OFF
    CONAN_RECIPE          gtest
    CONAN_FIND            GTest
    CONAN_VERSION         "1.12.1"
    CONAN_TARGETS         "GTest::gtest;GTest::gtest_main;GTest::gmock;GTest::gmock_main"
    VCPKG_FIND            GTest
    VCPKG_TARGETS         "GTest::gtest;GTest::gtest_main;GTest::gmock;GTest::gmock_main"
    PROVIDED_ACQUIRE      FETCHCONTENT
    PROVIDED_FC_REPO      "https://github.com/google/googletest.git"
    PROVIDED_FC_TAG       "release-1.12.1"
    PROVIDED_FC_TARGETS   "gtest;gtest_main;gmock;gmock_main"
)
