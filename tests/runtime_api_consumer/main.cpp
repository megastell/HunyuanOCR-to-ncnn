#include <hunyuanocr/runtime.h>

int main()
{
    hunyuanocr::RuntimeOptions options;
    options.use_packing_layout = false;
    hunyuanocr::Runtime runtime(options);
    return 0;
}
