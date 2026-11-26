idf_component_register(SRCS "main.c" "bench_core.c" INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -Os -fno-strict-aliasing -Wno-unused-function -Wno-unused-variable)
