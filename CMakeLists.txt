cmake_minimum_required(VERSION 3.16)
project(logq VERSION 1.0.0 LANGUAGES C)

# Header-only library: include/logq.h (all catalogued variants) or a header
# generated for one variant with tools/logqgen.py.
add_library(logq INTERFACE)
target_include_directories(logq INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(logq INTERFACE c_std_99)

option(LOGQ_BUILD_TESTS "Build tests and examples" ON)
option(LOGQ_EXHAUSTIVE "Register the exhaustive binary32 test (about 10 min on 2 cores)" OFF)

if(LOGQ_BUILD_TESTS)
  enable_testing()
  find_library(M_LIB m)
  find_package(OpenMP COMPONENTS C)
  set(CMAKE_C_STANDARD 11)

  foreach(t test_kernel test_special test_q31_ln test_float test_baselines test_r3_vs_aor)
    add_executable(${t} tests/${t}.c)
    target_link_libraries(${t} PRIVATE logq ${M_LIB})
    if(OpenMP_C_FOUND AND NOT (t STREQUAL "test_kernel" OR t STREQUAL "test_special"))
      target_link_libraries(${t} PRIVATE OpenMP::OpenMP_C)
    endif()
  endforeach()
  add_test(NAME kernel_exhaustive COMMAND test_kernel)
  add_test(NAME special_values COMMAND test_special)
  if(LOGQ_EXHAUSTIVE)
    add_test(NAME binary32_exhaustive COMMAND test_float)
    add_test(NAME q31_ln_vs_cmsis COMMAND test_q31_ln)
  endif()

  add_executable(db_meter examples/db_meter.c)
  target_link_libraries(db_meter PRIVATE logq ${M_LIB})
  add_test(NAME example_db_meter COMMAND db_meter)

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mylog.h
      COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tools/logqgen.py
              -n 2 -p 4 --seg g --prefix mylog_ -o ${CMAKE_CURRENT_BINARY_DIR}/mylog.h
      DEPENDS tools/logqgen.py)
    add_executable(custom_variant examples/custom_variant.c ${CMAKE_CURRENT_BINARY_DIR}/mylog.h)
    target_include_directories(custom_variant PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(custom_variant PRIVATE ${M_LIB})
    add_test(NAME example_custom_variant COMMAND custom_variant)
    add_test(NAME generator_reproducible
             COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_generated.py
                     ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  add_executable(bench_host bench/bench_host.c)
  target_link_libraries(bench_host PRIVATE logq ${M_LIB})
endif()
