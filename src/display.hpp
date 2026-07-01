#pragma once

#include "media.hpp"

#include <memory>
#include <string>

namespace rfdetr::media {

/// Live preview window for the --display flag. Presents BGR24 Image frames.
/// The backend (SDL2 or OpenCV HighGUI) is selected at compile time via the
/// CMake `USE_OPENCV` option. Degrades to a no-op (with a one-time warning) on
/// headless systems or when the window cannot be created, so the pipeline keeps
/// running regardless.
class Display {
  public:
    Display(const std::string &title, int width, int height);
    ~Display();

    Display(const Display &) = delete;
    Display &operator=(const Display &) = delete;
    Display(Display &&) = delete;
    Display &operator=(Display &&) = delete;

    /// Show one frame. Returns false when the user closed the window or pressed
    /// ESC/q (caller should stop the preview); returns true otherwise, including
    /// when the display is disabled (no-op).
    bool show(const Image &frame);

    /// True if a real window is backing this display.
    [[nodiscard]] bool ok() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rfdetr::media
