#pragma once

#include "media.hpp"

#include <SDL.h>
#include <string>

namespace rfdetr::media {

/// SDL2-backed live preview window for the --display flag. Presents BGR24 Image
/// frames. Degrades to a no-op (with a one-time warning) on headless systems or
/// when SDL cannot create a window, so the pipeline keeps running regardless.
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
    [[nodiscard]] bool ok() const noexcept { return ok_; }

  private:
    SDL_Window *window_{nullptr};
    SDL_Renderer *renderer_{nullptr};
    SDL_Texture *texture_{nullptr};
    int width_{0};
    int height_{0};
    bool ok_{false};
    bool quit_{false};
    bool warned_{false};
};

} // namespace rfdetr::media
