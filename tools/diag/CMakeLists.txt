
file(GLOB diagSources *.cpp)
file(GLOB diagIncludes *.h )

add_executable(deepRL-diagnostic ${diagSources})
target_link_libraries(deepRL-diagnostic jetson-reinforcement)
