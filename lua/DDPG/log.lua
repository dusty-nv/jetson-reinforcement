--[[

    Based on the code of Brendan Shillingford [bitbucket.org/bshillingford/nnob](https://bitbucket.org/bshillingford/nnob).

    Based on [github.com/rxi/log.lua](https://github.com/rxi/log.lua/commit/93cfbe0c91bced6d3d061a58e7129979441eb200).

    Usage:
    ```
    local log = require 'nnob.log'

    -- ...
    log.info('hello world!')
    log.infof('some number: %3.3f', 123)
    ```

    Pasted from README.md of `github.com/rxi/log.lua`:
    # log.lua
    A tiny logging module for Lua. 

    ![screenshot from 2014-07-04 19 55 55](https://cloud.githubusercontent.com/assets/3920290/3484524/2ea2a9c6-03ad-11e4-9ed5-a9744c6fd75d.png)


    ## Usage
    log.lua provides 6 functions, each function takes all its arguments,
    concatenates them into a string then outputs the string to the console and --
    if one is set -- the log file:

    * **log.trace(...)**
    * **log.debug(...)**
    * **log.info(...)**
    * **log.warn(...)**
    * **log.error(...)**
    * **log.fatal(...)**


    ### Additional options
    log.lua provides variables for setting additional options:

    #### log.usecolor
    Whether colors should be used when outputting to the console, this is `true` by
    default. If you're using a console which does not support ANSI color escape
    codes then this should be disabled.

    #### log.outfile
    The name of the file where the log should be written, log files do not contain
    ANSI colors and always use the full date rather than just the time. By default
    `log.outfile` is `nil` (no log file is used). If a file which does not exist is
    set as the `log.outfile` then it is created on the first message logged. If the
    file already exists it is appended to.

    #### log.level
    The minimum level to log, any logging function called with a lower level than
    the `log.level` is ignored and no text is outputted or written. By default this
    value is set to `"trace"`, the lowest log level, such that no log messages are
    ignored.

    The level of each log mode, starting with the lowest log level is as follows:
    `"trace"` `"debug"` `"info"` `"warn"` `"error"` `"fatal"`


    ## License
    This library is free software; you can redistribute it and/or modify it under
    the terms of the MIT license. See [LICENSE](LICENSE) for details.
--]]



--
-- log.lua
--
-- Copyright (c) 2016 rxi
--
-- This library is free software; you can redistribute it and/or modify it
-- under the terms of the MIT license. See LICENSE for details.
--

local log = { _version = "0.1.0" }

log.usecolor = true
log.outfile = nil
log.level = "trace"


local modes = {
    { name = "trace", color = "\27[34m", },
    { name = "debug", color = "\27[36m", },
    { name = "info", color = "\27[32m", },
    { name = "warn", color = "\27[33m", },
    { name = "error", color = "\27[31m", },
    { name = "fatal", color = "\27[35m", },
}


local levels = {}
for i, v in ipairs(modes) do
    levels[v.name] = i
end


local round = function(x, increment)
    increment = increment or 1
    x = x / increment
    return (x > 0 and math.floor(x + .5) or math.ceil(x - .5)) * increment
end


local _tostring = tostring

local tostring = function(...)
    local t = {}
    for i = 1, select('#', ...) do
        local x = select(i, ...)
        if type(x) == "number" then
            x = round(x, .01)
        end
        t[#t + 1] = _tostring(x)
    end
    return table.concat(t, " ")
end


for i, x in ipairs(modes) do
    local nameupper = x.name:upper()
    log[x.name] = function(...)

        -- Return early if we're below the log level
        if i < levels[log.level] then
            return
        end

        local msg = tostring(...)
        local info = debug.getinfo(2, "Sl")
        local lineinfo = info.short_src .. ":" .. info.currentline

        -- Output to console
        print(string.format("%s[%-6s%s]%s %s: %s",
            log.usecolor and x.color or "",
            nameupper,
            os.date("%H:%M:%S"),
            log.usecolor and "\27[0m" or "",
            lineinfo,
            msg))

        -- Output to log file
        if log.outfile then
            local fp = io.open(log.outfile, "a")
            local str = string.format("[%-6s%s] %s: %s\n",
                nameupper, os.date(), lineinfo, msg)
            fp:write(str)
            fp:close()
        end
    end

    -- bshillingford: add formatted versions,
    -- e.g. log.infof(...) as alias for log.info(string.format(...)
    log[x.name .. 'f'] = function(...)
        -- Return early if we're below the log level
        if i < levels[log.level] then
            return
        end

        local fmt = ... -- i.e. first arg; note: select(2, ...) gets everything after

        local info = debug.getinfo(2, "Sl")
        local lineinfo = info.short_src .. ":" .. info.currentline

        -- Output to console
        print(string.format("%s[%-6s%s]%s %s: " .. fmt,
            log.usecolor and x.color or "",
            nameupper,
            os.date("%H:%M:%S"),
            log.usecolor and "\27[0m" or "",
            lineinfo,
            select(2, ...)))

        -- Output to log file
        if log.outfile then
            local fp = io.open(log.outfile, "a")
            local str = string.format("[%-6s%s] %s: " .. fmt .. "\n",
                nameupper, os.date(), lineinfo, select(2, ...))
            fp:write(str)
            fp:close()
        end
    end
end


return log
