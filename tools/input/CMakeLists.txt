
file(GLOB inputSources *.cpp)
file(GLOB inputIncludes *.h )

add_executable(deepRL-input ${inputSources})
target_link_libraries(deepRL-input jetson-reinforcement)
