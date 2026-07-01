#include "display.hpp"

#include <iostream>

namespace rfdetr::media {

namespace {

void warn_once(bool &flag, const std::string &msg) {
    if (!flag) {
        flag = true;
        std::cerr << "warning: " << msg << std::endl;
    }
}

} // namespace

Display::Display(const std::string &title, int width, int height) : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
        warn_once(warned_, "Display: invalid dimensions, preview disabled");
        return;
    }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        warn_once(warned_, std::string("Display: SDL_Init failed (") + SDL_GetError() + "), preview disabled");
        return;
    }

    window_ = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width_, height_, 0);
    if (window_ == nullptr) {
        warn_once(warned_, std::string("Display: SDL_CreateWindow failed (") + SDL_GetError() + "), preview disabled");
        SDL_Quit();
        return;
    }

    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer_ == nullptr) {
        renderer_ = SDL_CreateRenderer(window_, -1, 0);
    }
    if (renderer_ == nullptr) {
        warn_once(warned_,
                  std::string("Display: SDL_CreateRenderer failed (") + SDL_GetError() + "), preview disabled");
        SDL_DestroyWindow(window_);
        window_ = nullptr;
        SDL_Quit();
        return;
    }

    texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, width_, height_);
    if (texture_ == nullptr) {
        warn_once(warned_, std::string("Display: SDL_CreateTexture failed (") + SDL_GetError() + "), preview disabled");
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
        SDL_DestroyWindow(window_);
        window_ = nullptr;
        SDL_Quit();
        return;
    }

    ok_ = true;
}

Display::~Display() {
    if (texture_ != nullptr) {
        SDL_DestroyTexture(texture_);
    }
    if (renderer_ != nullptr) {
        SDL_DestroyRenderer(renderer_);
    }
    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
    }
    if (ok_) {
        SDL_Quit();
    }
}

bool Display::show(const Image &frame) {
    if (!ok_ || quit_) {
        return !quit_;
    }
    if (frame.width != width_ || frame.height != height_) {
        // Size mismatch: skip this frame rather than tearing down the window.
        return true;
    }

    SDL_Event event;
    while (SDL_PollEvent(&event) != 0) {
        if (event.type == SDL_QUIT) {
            quit_ = true;
            return false;
        }
        if (event.type == SDL_KEYDOWN) {
            const auto key = event.key.keysym.sym;
            if (key == SDLK_ESCAPE || key == SDLK_q) {
                quit_ = true;
                return false;
            }
        }
    }

    SDL_UpdateTexture(texture_, nullptr, frame.data(), width_ * 3);
    SDL_RenderClear(renderer_);
    SDL_RenderCopy(renderer_, texture_, nullptr, nullptr);
    SDL_RenderPresent(renderer_);
    return true;
}

} // namespace rfdetr::media
