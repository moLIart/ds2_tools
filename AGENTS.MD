# Project

Reverse-engineering tools for You Are Empty's DS2 map and DS2MD model formats: Python parsers, format research, and C++20 Qt 6/QRhi viewers with animation playback and GLB export. C++ sources and headers are in `viewer/`; build with `build.ps1` using Qt 6.11, MSVC x64, CMake, and Ninja.

## C++ block spacing

- Add one empty line after completed control-flow blocks (`if`/`else`, `for`, `while`, `do`/`while`, `switch`, and `try`/`catch`) before the next statement. Omit the empty line when the block is the final statement in its enclosing scope, not just at the end of a function.
- Keep continuation clauses attached: do not insert an empty line before `else`, `catch`, or the terminating `while` of a `do`/`while` statement.
