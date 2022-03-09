
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <GL/glew.h>

#include "program.h"
#include "gpu_texture.cuh"
#include "mandelbrot_kernel.cuh"

namespace FastMandelbrot
{

    program::program(unsigned int width, unsigned int height)
        : _width{width}, _height{height}
    {
        SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");
        SDL_Init(SDL_INIT_VIDEO);

        _window = SDL_CreateWindow(
            "Fast Mandelbrot", 0, 0,
            width, height,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

        // Hide the windows until the rendering is not ready to start
        SDL_HideWindow(_window);

        _gl_context = SDL_GL_CreateContext(_window);

        GLenum glew_error = glewInit();
        if (glew_error != GLEW_OK)
        {
            throw std::runtime_error(reinterpret_cast<const char *>(glewGetErrorString(glew_error)));
        }

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

        // Create openGL texture to render into
        glGenTextures(1, &_texture_id);
        glBindTexture(GL_TEXTURE_2D, _texture_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glEnable(GL_TEXTURE_2D);
    }

    program::~program() noexcept
    {
        glDeleteTextures(1, &_texture_id);
        SDL_GL_DeleteContext(_gl_context);
        SDL_DestroyWindow(_window);
    }

    void program::run()
    {
        registered_texture texture{_texture_id, _width, _height};

        SDL_ShowWindow(_window);
        _update_size();

        while (_handle_events(texture));

        SDL_HideWindow(_window);
    }

    bool program::_handle_key_down(SDL_Keysym key)
    {
        switch (key.sym)
        {
            case SDLK_KP_PLUS:
                _step_count *= 2;
                break;

            case SDLK_KP_MINUS:
                _step_count /= 2;
                if (_step_count < 2) _step_count = 1;
                break;

            case SDLK_RETURN:
                break;

            case SDLK_SPACE:
                break;

            default:
                return false;
        }

        std::cout << "step count = " << _step_count << std::endl;
        return true;
    }

    bool program::_handle_mouse_wheel(bool up)
    {
        constexpr auto factor = 1.1f;
        const auto new_size = _size * (up ? (1.f / factor) : factor);
        const auto origin_offset = (_size - new_size) / 2.f;

        _size = new_size;
        _origin_x += origin_offset;
        _origin_y += origin_offset;

        return true;
    }

    bool program::_handle_mouse_drag(int xrel, int yrel)
    {
        const auto unit_per_pixel = _size / static_cast<float>(_width);
        _origin_x -= unit_per_pixel * static_cast<float>(xrel);
        _origin_y += unit_per_pixel * static_cast<float>(yrel);
        return true;
    }

    bool program::_handle_events(registered_texture& texture)
    {
        SDL_Event event;
        bool redraw = false;

        while (SDL_PollEvent(&event)) {
            switch (event.type)
            {
                case SDL_KEYDOWN:
                    redraw = _handle_key_down(event.key.keysym);
                    break;
                case SDL_MOUSEWHEEL:
                    redraw = _handle_mouse_wheel(event.wheel.y > 0);
                    break;
                case SDL_MOUSEMOTION:
                    if (_drag)
                        redraw = _handle_mouse_drag(event.motion.xrel, event.motion.yrel);
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT) _drag = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT) _drag = false;
                    break;
                case SDL_WINDOWEVENT:
                    if (event.window.event == SDL_WINDOWEVENT_RESIZED)
                        _update_size();
                    else if (event.window.event == SDL_WINDOWEVENT_EXPOSED)
                        redraw = true;
                    break;
                case SDL_QUIT:
                    return false;
            }
        }

        if (redraw)
        {
            using namespace std::chrono;

            const auto start = steady_clock::now();
            _render_frame(texture);
            const auto end = steady_clock::now();
            const auto duration = duration_cast<microseconds>(end - start).count();
            const auto fps = static_cast<int>(1.E6 / static_cast<double>(duration));
            std::cout << fps << " fps (" << (duration / 1000u) << " ms)\n";
            _draw_texture();
        }

        return true;
    }

    void program::_draw_texture()
    {
        // Draw a quad with the texture on it (ugly old shoold open gl...)
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, _texture_id);

        glBegin(GL_QUADS);

        glVertex2i(0, 0);
        glTexCoord2i(1, 0);

        glVertex2i(1, 0);
        glTexCoord2i(1, 1);

        glVertex2i(1, 1);
        glTexCoord2i(0, 1);

        glVertex2i(0, 1);
        glTexCoord2i(0, 0);

        glEnd();

        // double buffering
        SDL_GL_SwapWindow(_window);
    }

    void program::_update_size()
    {
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(_window, &width, &height);
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.f, 1.f, 0.f, 1.f, -1.f, 1.f);
    }

    void program::_render_frame(registered_texture& texture)
    {
        const auto origin = float2{_origin_x, _origin_y};
        auto mapped_surface = texture.get_mapped_surface();
        call_mandelbrot_kernel(mapped_surface.surface(), _width, _height, origin, _size, _step_count);
    }
}
