
-- this is a one-line comment, beginning after the double-dash
print('HELLO from LUA!')


-- variables
myVar = 16.0	-- variables are floating-point doubles
local myVar2 = 32.0

print('my variable equals ' .. myVar)


-- tables
list = {1, 2, 3}
map  = {x = 10, y = 20, z = 30}

print('list  ' .. list[1])	-- indexing starts with 1
print('map.x ' .. map.x)


-- functions
function mult(a, b)
	return a * b
end

print('mult ' .. mult(map.x, map.y))


-- we're done here
print('goodbye!')

