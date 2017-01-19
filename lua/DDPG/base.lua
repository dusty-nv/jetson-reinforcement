local tools = {}

function tools.dc(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[tools.dc(orig_key)] = tools.dc(orig_value)
        end
        setmetatable(copy, tools.dc(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

return tools