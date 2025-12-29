cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(openlift_split_core)

target_sources(app PRIVATE src/main.c src/cia417_mock.c)
target_include_directories(app PRIVATE src)

