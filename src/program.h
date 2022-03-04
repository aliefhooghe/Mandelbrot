#ifndef PROGRAM_H_
#define PROGRAM_H_

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

namespace FastMandelbrot
{
    class registered_texture;

    class program
    {

    public:
        program(unsigned int width, unsigned int height);
        program(const program&) = delete;
        program(program&&) noexcept = delete;
        ~program() noexcept;

        void run();

    private:
        void _handle_key_down(SDL_Keysym key);
        void _handle_mouse_wheel(bool up);
        void _handle_mouse_motion(int xrel, int yrel);
        bool _handle_events();
        void _draw_texture();
        void _update_size();

        void _render_frame(registered_texture&);

        const unsigned int _width;
        const unsigned int _height;
        SDL_Window *_window;
        SDL_GLContext _gl_context;
        GLuint _texture_id;

        float _origin_x{0.f};
        float _origin_y{0.f};
        float _radius{1.5f};
        unsigned int _step_count{100};
    };

}

#endif /* MANDELBROT_RENDERER_H_ */
