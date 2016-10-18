
-- this is a one-line comment, beginning after the double-dash
print('testing LUAJIT51 scripting language')

--[[ this is
	a multi-line
	block comment --]]

-- variables
myVar = 16.0	-- variables are floating-point doubles
local myVar2 = 32.0

print('my variable equals ' .. myVar)


-- tables
list = {1, 2, 3}
map  = {x = 10, y = 20, z = 30}

print('list  ' .. list[1])	-- indexing starts with 1
print('map.x ' .. map.x)


-- loops and branches
for i=1,10 do
	if i == 1 then
		print("one")
	elseif i == 2 then
		print("two")
	else
		print(i)
	end
end


-- functions
function multiply(a, b)
	return a * b
end

multiplyResult = multiply(map.x, map.y)

print('multiply = ' .. multiplyResult)


-- we're done here
print('goodbye!')


