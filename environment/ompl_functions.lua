local lfs = require("lfs")
full_path = lfs.currentdir()
path = ";"..string.sub(full_path, 1, -5).."baseline/baselines/ppo1/environment/?.lua"
--print(path)
package.path=package.path .. path

require("common_functions")
require("robot_control")
require("get_set")

--------------------------------------------- ompl functions -------------------------------------------------------
_robot_hd = simGetObjectHandle('centauro')
_robot_dim = 2
_joint_dim = 4
_collision_hd_1 = nil
_collision_hd_2 = nil
_use_validation_callback = false

_path = {}

init_statespace=function(robot_hd, joint_hds, start_pose, goal_pose)
    local state_spaces={}

    local min_y = start_pose[2]-1.5
    local max_y = start_pose[2]+1.5

    local min_x = start_pose[1]-1.5
    local max_x = start_pose[1]+1.5

    local min_z = 0
    local max_z = 0

    if _robot_dim > 1 then
        min_z = start_pose[3] - 0.1
        max_z = start_pose[3] + 0.1
    end
    
    if _robot_dim == 1 then
        min_range={min_x,min_y}
        max_range={max_x,max_y}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_position2d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 2 then
        min_range={min_x,min_y}
        max_range={max_x,max_y}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_pose2d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 3 then
        min_range={min_x,min_y, min_z}
        max_range={max_x,max_y, max_z}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_position3d,robot_hd,min_range,max_range,weight_move)               -- base space
    elseif _robot_dim == 4 then
        min_range={min_x,min_y, min_z}
        max_range={max_x,max_y, max_z}
        local weight_move = 3
        state_spaces[1]=simExtOMPL_createStateSpace('base_space',sim_ompl_statespacetype_pose3d,robot_hd,min_range,max_range,weight_move)               -- base space        
    end

    -- height
    min_range={-0.15}
    max_range={0.15}
    local weight_move = 3
    state_spaces[2]=simExtOMPL_createStateSpace('height',sim_ompl_statespacetype_joint_position,robot_hd,min_range,max_range,weight_move) 

    -- leg
    min_range={-0.0}
    max_range={0.15}
    local weight_move = 3
    state_spaces[3]=simExtOMPL_createStateSpace('leg_width',sim_ompl_statespacetype_joint_position,robot_hd,min_range,max_range,weight_move) 

    -- for i=1,_joint_dim,1 do
    --     local cyclic, range = simGetJointInterval(joint_hds[i])
    --     if cyclic == true then 
    --         range = {-math.pi, math.pi}
    --     end
    --     local weight = 1
    --     print(cyclic, range[1], range[2])     
    --     state_spaces[#state_spaces+1]=simExtOMPL_createStateSpace('joint'..i,sim_ompl_statespacetype_joint_position,joint_hds[i],{range[1]},{range[2]}, weight,robot_hd)
    -- end

    return state_spaces
end


init_params=function(robot_dim, joint_dim, collision_name1, collision_name2, use_validation_callback)
    set_robot_dim(robot_dim)
    set_joint_dim(joint_dim)
    set_collision_hd(collision_name1, collision_name2)
    set_use_callback(use_validation_callback)
end

init_task=function(start_name, task_id)
    local robot_hd       =simGetObjectHandle(start_name)
    local target_hd      =simGetObjectHandle('target')
    local joint_hds      =get_joint_hds(_joint_dim)

    local task_hd = simExtOMPL_createTask(task_id)
    -- simExtOMPL_setVerboseLevel(task_hd, 3)
    simExtOMPL_setAlgorithm(task_hd,sim_ompl_algorithm_RRTConnect)
    -- simExtOMPL_setAlgorithm(task_hd,sim_ompl_algorithm_pSBL)

    ------ callbacks ---------------\
    simExtOMPL_setGoalCallback(task_hd, 'goalSatisfied')
    if _use_validation_callback then
        simExtOMPL_setStateValidationCallback(task_hd, 'stateValidation')
    end

    -- -- start pose --
    startpose = get_state_hl(robot_hd, _robot_dim) 
    goalpose = get_state_hl(target_hd, _robot_dim)

    -- print ('start pose ', #startpose, #joint_hds, _robot_dim)

    satat_spaces=init_statespace(robot_hd, joint_hds, startpose, goalpose) -- for sample state
    simExtOMPL_setStateSpace(task_hd, satat_spaces)

    simExtOMPL_setCollisionPairs(task_hd, {_collision_hd_1, _collision_hd_2}) -- collision 

    simExtOMPL_setStartState(task_hd, startpose)    
    simExtOMPL_setGoalState(task_hd, goalpose)

    -- -- simExtOMPL_printTaskInfo(task_hd)
    return task_hd, #startpose
end

compute_path=function(task_hd, max_time)
    -- forbidThreadSwitches(true)
    r,_path=simExtOMPL_compute(task_hd, max_time, -1, 50)
    -- forbidThreadSwitches(false)

    path_step = #_path/#startpose
    --local txt='finish compute' ..path_step
    --displayInfo(txt)
    
    return _path
end

set_robot_dim=function(robot_dim)
    _robot_dim = robot_dim
end

set_joint_dim=function(joint_dim)
    _joint_dim = joint_dim
end

set_collision_hd=function(name1, name2)
    _collision_hd_1 = simGetCollectionHandle(name1)
    _collision_hd_2 = simGetCollectionHandle(name2)
end

set_use_callback=function(use_validation_callback, function_name)
    _use_validation_callback = use_validation_callback
end

goalSatisfied = function(state)
    local satisfied=0
    local dist=0
    local diff={}
    for i=1, #goalpose, 1 do
        diff[i]=math.abs(state[i]-goalpose[i])
    end

    local min_dist = 0.2
    if diff[1] < min_dist and diff[2] < min_dist then
    -- if state[1]-_callback_goal[1] < 0.05 and state[2]-_callback_goal[2] < 0.1 then
        satisfied=1
    end

    dist= math.sqrt(diff[1]*diff[1] + diff[2]*diff[2]) --diff[3]+diff[4]+diff[5]+diff[6]+diff[7]
    return satisfied, dist
end


stateValidation=function(state)
    local is_valid = false
    -- forbidThreadSwitches(true)
    -- print('validation!!!')
    local pass = do_action_hl(_robot_hd, state)
    if pass == 't' then
        is_valid = true
    end
    --res = simExtOMPL_writeState(_task_hd, current_state)
    -- sleep(1)
    -- simSwitchThread()
    -- forbidThreadSwitches(false)
    return is_valid
end
