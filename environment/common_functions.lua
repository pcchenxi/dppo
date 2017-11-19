
------------------------------------------ system functions ----------------------------------
function sleep(s)
  local ntime = os.clock() + s
  repeat until os.clock() > ntime
end

function print_table(table)
    for i=1, #table, 1 do
        print(table[i]..' ')
    end
end


function copy_table(obj)
  if type(obj) ~= 'table' then return obj end
  local res = {}
  for k, v in pairs(obj) do res[copy_table(k)] = copy_table(v) end
  return res
end

displayInfo=function(txt)
    if dlgHandle then
        simEndDialog(dlgHandle)
    end
    dlgHandle=nil
    if txt and #txt>0 then
        dlgHandle=simDisplayDialog('info',txt,sim_dlgstyle_message,false)
        simSwitchThread()
    end
end

forbidThreadSwitches=function(forbid)
    -- Allows or forbids automatic thread switches.
    -- This can be important for threaded scripts. For instance,
    -- you do not want a switch to happen while you have temporarily
    -- modified the robot configuration, since you would then see
    -- that change in the scene display.
    local forbidLevel = 0
    if forbid then
        forbidLevel=forbidLevel+1
        if forbidLevel==1 then
            simSetThreadAutomaticSwitch(false)
        end
    else
        forbidLevel=forbidLevel-1
        if forbidLevel==0 then
            simSetThreadAutomaticSwitch(true)
        end
    end
end

