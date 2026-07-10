deps_declare(FFmpeg
    REQUIRED             OFF
    APT                  ON
    APT_METHOD           PKG_CONFIG
    APT_PKG_PREFIX       FFMPEG
    APT_PKG_MODULES      "libavcodec;libavformat;libavutil;libswscale"
    APT_IMPORTED_TARGETS "PkgConfig::FFMPEG"
    VCPKG_FIND           FFMPEG
    VCPKG_TARGETS        "FFMPEG::avcodec;FFMPEG::avformat;FFMPEG::avutil;FFMPEG::swscale"
    CONAN_FIND           FFmpeg
    CONAN_TARGETS        "FFmpeg::avcodec;FFmpeg::avformat;FFmpeg::avutil;FFmpeg::swscale"
)
