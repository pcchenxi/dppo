-- Check the end of the script for some explanations!
sum_acc
count = 0

function get_acc_data(inInts,inFloats,inStrings,inBuffer)
    local sum_acc = sum_acc/count

    sum_acc = 0
    count = 0
    return {}, {sum_acc}, {}, ''
end

if (sim_call_type==sim_childscriptcall_initialization) then 
    modelBase=simGetObjectAssociatedWithScript(sim_handle_self)
    massObject=simGetObjectHandle('Accelerometer_mass')
    sensor=simGetObjectHandle('Accelerometer_forceSensor')
    result,mass=simGetObjectFloatParameter(massObject,sim_shapefloatparam_mass)
    ui=simGetUIHandle('Accelerometer_UI')
    simSetUIButtonLabel(ui,0,simGetObjectName(modelBase))
    accelCommunicationTube=simTubeOpen(0,'accelerometerData'..simGetNameSuffix(nil),1)
end 

if (sim_call_type==sim_childscriptcall_cleanup) then 
 
end 

if (sim_call_type==sim_childscriptcall_sensing) then 
    result,force=simReadForceSensor(sensor)
    if (result>0) then
        accel={force[1]/mass,force[2]/mass,force[3]/mass}
        simTubeWrite(accelCommunicationTube,simPackFloatTable(accel))
        simSetUIButtonLabel(ui,3,string.format("X-Accel: %.4f",accel[1]))
        simSetUIButtonLabel(ui,4,string.format("Y-Accel: %.4f",accel[2]))
        simSetUIButtonLabel(ui,5,string.format("Z-Accel: %.4f",accel[3]))
        local acc = math.sqrt(accel[1]*accel[1] + accel[2]*accel[2] + accel[3]*accel[3]) - 9.8
        sum_acc = sum_acc + acc
        count = count + 1
    else
        simSetUIButtonLabel(ui,3,"X-Accel: -")
        simSetUIButtonLabel(ui,4,"Y-Accel: -")
        simSetUIButtonLabel(ui,5,"Z-Accel: -")
    end
    
    -- To read data from this accelerometer in another script, use following code:
    --
    -- accelCommunicationTube=simTubeOpen(0,'accelerometerData'..simGetNameSuffix(nil),1) -- put this in the initialization phase
    -- data=simTubeRead(accelCommunicationTube)
    -- if (data) then
    --     acceleration=simUnpackFloatTable(data)
    -- end
    --
    -- If the script in which you read the acceleration has a different suffix than the accelerometer suffix,
    -- then you will have to slightly adjust the code, e.g.:
    -- accelCommunicationTube=simTubeOpen(0,'accelerometerData#') -- if the accelerometer script has no suffix
    -- or
    -- accelCommunicationTube=simTubeOpen(0,'accelerometerData#0') -- if the accelerometer script has a suffix 0
    -- or
    -- accelCommunicationTube=simTubeOpen(0,'accelerometerData#1') -- if the accelerometer script has a suffix 1
    -- etc.
    --
    --
    -- You can of course also use global variables (not elegant and not scalable), e.g.:
    -- In the accelerometer script:
    -- simSetFloatSignal('accelerometerX',accel[1])
    -- simSetFloatSignal('accelerometerY',accel[2])
    -- simSetFloatSignal('accelerometerZ',accel[3])
    --
    -- And in the script that needs the data:
    -- xAccel=simGetFloatSignal('accelerometerX')
    -- yAccel=simGetFloatSignal('accelerometerY')
    -- zAccel=simGetFloatSignal('accelerometerZ')
    --
    -- In addition to that, there are many other ways to have 2 scripts exchange data. Check the documentation for more details
end 
