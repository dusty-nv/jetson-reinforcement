

file(GLOB fruitSources *.cpp)
file(GLOB fruitIncludes *.h )

add_executable(fruit ${fruitSources})
target_link_libraries(fruit jetson-reinforcement)
