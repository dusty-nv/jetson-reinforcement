
file(GLOB diagSources *.cpp)
file(GLOB diagIncludes *.h )

add_executable(deepRL-diag ${diagSources})
target_link_libraries(deepRL-diag jetson-reinforcement)
