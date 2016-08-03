
file(GLOB consoleSources *.cpp)
file(GLOB consoleIncludes *.h )

add_executable(deepRL-console ${consoleSources})
target_link_libraries(deepRL-console jetson-reinforcement)
