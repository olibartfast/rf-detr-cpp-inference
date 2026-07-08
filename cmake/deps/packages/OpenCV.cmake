deps_declare(OpenCV
    REQUIRED            OFF
    DEFINITIONS         USE_OPENCV
    APT                 ON
    APT_METHOD          FIND_PACKAGE
    APT_FIND_NAME       OpenCV
    APT_COMPONENTS      "core;imgcodecs;imgproc;videoio;highgui"
    APT_LINK_VARS       "OpenCV_LIBS"
    APT_INCLUDE_VAR     "OpenCV_INCLUDE_DIRS"
    APT_VERSION_VAR     "OpenCV_VERSION"
    CONAN_RECIPE        opencv
    CONAN_FIND          opencv
    CONAN_TARGETS       "opencv::opencv"
    VCPKG_FIND          OpenCV
    VCPKG_TARGETS       "opencv_core;opencv_imgcodecs;opencv_imgproc;opencv_videoio;opencv_highgui"
)
