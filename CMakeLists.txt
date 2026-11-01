cmake_minimum_required(VERSION 3.24)

project(ds2_viewers VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.11 REQUIRED COMPONENTS Widgets GuiPrivate ShaderTools)

qt_standard_project_setup()

add_library(viewer_common STATIC
    viewer/scene.h viewer/camera.h
    viewer/map_loader.cpp viewer/model_loader.cpp viewer/model_animation.cpp
    viewer/material.cpp viewer/material.h
    viewer/model_animation.h viewer/texture.cpp viewer/texture.h
    viewer/glb_export.h viewer/glb_export.cpp viewer/glb_animation.cpp
    viewer/render_widget.cpp viewer/render_widget.h viewer/viewer_window.cpp viewer/viewer_window.h)
target_link_libraries(viewer_common PUBLIC Qt6::Widgets Qt6::GuiPrivate)

if(WIN32)
    target_link_libraries(viewer_common PRIVATE user32)
endif()

if(MSVC)
    target_compile_options(viewer_common PRIVATE /W4 /utf-8)
else()
    target_compile_options(viewer_common PRIVATE -Wall -Wextra -Wpedantic)
endif()

qt_add_shaders(viewer_common viewer_shaders
    PREFIX "/shaders" BASE "viewer/shaders"
    GLSL "330,300es" HLSL 50 MSL 12
    FILES viewer/shaders/scene.vert viewer/shaders/scene.frag)

# Qt deployment requires an absolute prefix, even with a relative --prefix.
install(CODE [[
    cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_PREFIX NORMALIZE)
]])

foreach(viewer IN ITEMS ds2_view ds2md_view)
    qt_add_executable(${viewer} viewer/main.cpp)
    target_link_libraries(${viewer} PRIVATE viewer_common)
    
	if(viewer STREQUAL "ds2_view")
        target_compile_definitions(${viewer} PRIVATE DS2_MAP_VIEWER=1)
    endif()
    
	install(TARGETS ${viewer}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        BUNDLE DESTINATION .)
    
	if(WIN32)
        add_custom_command(TARGET ${viewer} POST_BUILD
            COMMAND "${QT6_INSTALL_PREFIX}/bin/windeployqt.exe" --no-translations --no-opengl-sw "$<TARGET_FILE:${viewer}>"
            COMMENT "Deploying Qt runtime for ${viewer}" VERBATIM)
        qt_generate_deploy_app_script(
            TARGET ${viewer}
            OUTPUT_SCRIPT deploy_script
            NO_TRANSLATIONS
            DEPLOY_TOOL_OPTIONS --no-opengl-sw)
        install(SCRIPT "${deploy_script}")
    endif()
endforeach()
