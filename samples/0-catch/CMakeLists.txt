

file(GLOB catchSources *.cpp)
file(GLOB catchIncludes *.h )

add_executable(catch ${catchSources})
target_link_libraries(catch jetson-reinforcement)
