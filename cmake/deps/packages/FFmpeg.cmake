deps_declare(FFmpeg
    REQUIRED             OFF
    APT                  ON
    APT_METHOD           PKG_CONFIG
    APT_PKG_PREFIX       FFMPEG
    APT_PKG_MODULES      "libavcodec;libavformat;libavutil;libswscale"
    APT_IMPORTED_TARGETS "PkgConfig::FFMPEG"
    CONAN_RECIPE         ffmpeg
    CONAN_FIND           ffmpeg
    CONAN_TARGETS        "ffmpeg::ffmpeg"
    VCPKG_FIND           FFMPEG
    VCPKG_TARGETS        "FFMPEG::avcodec;FFMPEG::avformat;FFMPEG::avutil;FFMPEG::swscale"
)
