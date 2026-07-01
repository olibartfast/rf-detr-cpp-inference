#include "display.hpp"

#include <iostream>
#include <string>

#ifdef USE_OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#else
#include <SDL.h>
#endif

namespace rfdetr::media {

namespace {

void warn_once(bool &flag, const std::string &msg) {
    if (!flag) {
        flag = true;
        std::cerr << "warning: " << msg << '\n';
    }
}

} // namespace

#ifdef USE_OPENCV

struct Display::Impl {
    std::string title;
    int width{0};
    int height{0};
    bool ok{false};
    bool quit{false};
    bool warned{false};

    Impl(std::string t, int w, int h) : title(std::move(t)), width(w), height(h) {
        if (width <= 0 || height <= 0) {
            warn_once(warned, "Display: invalid dimensions, preview disabled");
            return;
        }
        try {
            cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
            cv::resizeWindow(title, width, height);
            ok = true;
        } catch (const cv::Exception &e) {
            warn_once(warned,
                      std::string("Display: OpenCV window creation failed (") + e.what() + "), preview disabled");
        }
    }

    ~Impl() {
        if (ok) {
            try {
                cv::destroyWindow(title);
            } catch (const cv::Exception &) {
                // Ignore teardown errors.
            }
        }
    }

    bool show(const Image &frame) {
        if (!ok || quit) {
            return !quit;
        }
        try {
            cv::Mat mat(frame.height, frame.width, CV_8UC3, const_cast<uint8_t *>(frame.data()));
            cv::imshow(title, mat);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') {
                quit = true;
                return false;
            }
            return true;
        } catch (const cv::Exception &e) {
            warn_once(warned, std::string("Display: imshow failed (") + e.what() + "), preview disabled");
            ok = false;
            return true;
        }
    }
};

#else // SDL2 backend

struct Display::Impl {
    SDL_Window *window{nullptr};
    SDL_Renderer *renderer{nullptr};
    SDL_Texture *texture{nullptr};
    int width{0};
    int height{0};
    bool ok{false};
    bool quit{false};
    bool warned{false};

    Impl(const std::string &title, int w, int h) : width(w), height(h) {
        if (width <= 0 || height <= 0) {
            warn_once(warned, "Display: invalid dimensions, preview disabled");
            return;
        }

        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            warn_once(warned, std::string("Display: SDL_Init failed (") + SDL_GetError() + "), preview disabled");
            return;
        }

        window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
        if (window == nullptr) {
            warn_once(warned,
                      std::string("Display: SDL_CreateWindow failed (") + SDL_GetError() + "), preview disabled");
            SDL_Quit();
            return;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (renderer == nullptr) {
            renderer = SDL_CreateRenderer(window, -1, 0);
        }
        if (renderer == nullptr) {
            warn_once(warned,
                      std::string("Display: SDL_CreateRenderer failed (") + SDL_GetError() + "), preview disabled");
            SDL_DestroyWindow(window);
            window = nullptr;
            SDL_Quit();
            return;
        }

        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, width, height);
        if (texture == nullptr) {
            warn_once(warned,
                      std::string("Display: SDL_CreateTexture failed (") + SDL_GetError() + "), preview disabled");
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
            SDL_DestroyWindow(window);
            window = nullptr;
            SDL_Quit();
            return;
        }

        ok = true;
    }

    ~Impl() {
        if (texture != nullptr) {
            SDL_DestroyTexture(texture);
        }
        if (renderer != nullptr) {
            SDL_DestroyRenderer(renderer);
        }
        if (window != nullptr) {
            SDL_DestroyWindow(window);
        }
        if (ok) {
            SDL_Quit();
        }
    }

    bool show(const Image &frame) {
        if (!ok || quit) {
            return !quit;
        }
        if (frame.width != width || frame.height != height) {
            // Size mismatch: skip this frame rather than tearing down the window.
            return true;
        }

        SDL_Event event;
        while (SDL_PollEvent(&event) != 0) {
            if (event.type == SDL_QUIT) {
                quit = true;
                return false;
            }
            if (event.type == SDL_KEYDOWN) {
                const auto key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE || key == SDLK_q) {
                    quit = true;
                    return false;
                }
            }
        }

        SDL_UpdateTexture(texture, nullptr, frame.data(), width * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        return true;
    }
};

#endif // USE_OPENCV

Display::Display(const std::string &title, int width, int height)
    : impl_(std::make_unique<Impl>(title, width, height)) {}

Display::~Display() = default;

bool Display::show(const Image &frame) { return impl_->show(frame); }

bool Display::ok() const noexcept { return impl_->ok; }

} // namespace rfdetr::media
