# Builds cddlib's two libraries (float-only `cdd`, GMP-exact `cddgmp`) for
# pycddlib to link against. cddlib itself only ships GNU autotools, which
# has no MSVC support, so this mechanically replicates its own
# lib-src/Makefile.am + lib-src/Makefile.gmp.am recipe instead. GMP is
# expected to come from vcpkg -- configure with
# -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake (see
# build.py, which drives this end to end).
cmake_minimum_required(VERSION 3.21)
project(cddlib_vendor C)

# Both libraries below get linked into pycddlib's own shared object
# (cdd/__init__*.so) afterward, not just used as static archives on their
# own -- on Linux/gcc that fails at link time ("relocation R_X86_64_TPOFF32
# ... local-exec is incompatible with -shared") unless they're themselves
# built as position-independent code. No-op on MSVC (which has no such
# distinction), so safe to set unconditionally.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CDDLIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cddlib)
set(LIB_SRC ${CDDLIB_SRC}/lib-src)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(PATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patches)

# Apply the local incremental-append patch to the pristine submodule
# checkout. Idempotent (see apply_patch.py), so a plain reconfigure is
# safe to re-run.
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patch.py
          ${PATCH_DIR}/cddlib-incremental-append.patch ${CDDLIB_SRC}
  RESULT_VARIABLE _cddlib_patch_result
)
if(NOT _cddlib_patch_result EQUAL 0)
  message(FATAL_ERROR "Failed to apply cddlib-incremental-append.patch to ${CDDLIB_SRC}")
endif()

set(CDD_COMMON_SOURCES
  ${LIB_SRC}/cddcore.c
  ${LIB_SRC}/cddlp.c
  ${LIB_SRC}/cddmp.c
  ${LIB_SRC}/cddio.c
  ${LIB_SRC}/cddlib.c
  ${LIB_SRC}/cddproj.c
  ${LIB_SRC}/setoper.c
)

# --- libcdd: cddlib's plain (non-GMP) arithmetic backend -- autotools'
#     `-UGMPRATIONAL`, i.e. GMPRATIONAL simply never defined here. ---
add_library(cdd STATIC ${CDD_COMMON_SOURCES})
target_include_directories(cdd PUBLIC $<BUILD_INTERFACE:${LIB_SRC}>)

# --- GMP, provided by vcpkg. GMP ships no CMake package config of its
#     own, so locate it the standard way for such ports; the vcpkg
#     toolchain file adds its install tree to CMAKE_PREFIX_PATH. ---
find_path(GMP_INCLUDE_DIR NAMES gmp.h)
find_library(GMP_LIBRARY NAMES gmp)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
  message(FATAL_ERROR
    "GMP not found. Configure with "
    "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake "
    "(see mga-surveyor/vendor/cdd/build.py, which sets this up automatically).")
endif()

# --- Generate the "_f" (float/myfloat) variant sources/headers that
#     cddlib's own autotools build produces via sed+awk
#     (lib-src/Makefile.gmp.am's SED_GMP) -- see cmake/gen_f_variant.py
#     for the portable reimplementation, verified byte-identical to the
#     real sed/awk output. ---
set(GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_f_variant.py)
set(F_C_STEMS cddcore cddlp cddmp cddio cddlib cddproj)
set(F_H_STEMS cdd cddmp cddtypes)

set(GENERATED_C "")
foreach(stem IN LISTS F_C_STEMS)
  add_custom_command(
    OUTPUT ${GEN_DIR}/${stem}_f.c
    COMMAND ${Python3_EXECUTABLE} ${GEN_SCRIPT} ${LIB_SRC}/${stem}.c ${GEN_DIR}/${stem}_f.c
    DEPENDS ${LIB_SRC}/${stem}.c ${GEN_SCRIPT}
    VERBATIM
  )
  list(APPEND GENERATED_C ${GEN_DIR}/${stem}_f.c)
endforeach()

set(GENERATED_H "")
foreach(stem IN LISTS F_H_STEMS)
  add_custom_command(
    OUTPUT ${GEN_DIR}/${stem}_f.h
    COMMAND ${Python3_EXECUTABLE} ${GEN_SCRIPT} ${LIB_SRC}/${stem}.h ${GEN_DIR}/${stem}_f.h
    DEPENDS ${LIB_SRC}/${stem}.h ${GEN_SCRIPT}
    VERBATIM
  )
  list(APPEND GENERATED_H ${GEN_DIR}/${stem}_f.h)
endforeach()

add_custom_target(cddlib_f_variants DEPENDS ${GENERATED_C} ${GENERATED_H})

# --- libcddgmp: the real GMP-exact dd_* API (the same 7 sources,
#     recompiled with GMPRATIONAL defined) PLUS the redundant float-only
#     ddf_* API (the generated _f.c files) side by side in one library --
#     exactly upstream's libcddgmp.la. Defining GMPRATIONAL for the whole
#     target, generated sources included, is correct: gen_f_variant.py's
#     rename turns the literal token GMPRATIONAL into ddf_GMPRATIONAL
#     inside them, so their own #ifdef GMPRATIONAL guards never see it. ---
add_library(cddgmp STATIC ${CDD_COMMON_SOURCES} ${GENERATED_C})
add_dependencies(cddgmp cddlib_f_variants)
target_compile_definitions(cddgmp PRIVATE GMPRATIONAL)
target_include_directories(cddgmp PUBLIC
  $<BUILD_INTERFACE:${LIB_SRC}>
  $<BUILD_INTERFACE:${GEN_DIR}>
)
target_include_directories(cddgmp PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(cddgmp PRIVATE ${GMP_LIBRARY})

install(TARGETS cdd cddgmp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
# Under include/cddlib/, not include/ directly: pycddlib's own Cython sources
# #include "cddlib/cdd.h" etc (cdd.pxi), matching upstream cddlib's own
# autotools `make install` layout -- this has to mirror that exactly, since
# pycddlib's setup.py (and its Cython sources) are unmodified.
install(FILES
  ${LIB_SRC}/cdd.h
  ${LIB_SRC}/cddmp.h
  ${LIB_SRC}/cddtypes.h
  ${LIB_SRC}/setoper.h
  ${LIB_SRC}/splitmix64.h
  DESTINATION include/cddlib
)
# Generated at configure/build time (not yet on disk when install() below is
# parsed), so install them by directory rather than by static FILES list.
install(DIRECTORY ${GEN_DIR}/ DESTINATION include/cddlib FILES_MATCHING PATTERN "*_f.h")
