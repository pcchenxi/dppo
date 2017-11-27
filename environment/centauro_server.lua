local lfs = require("lfs")
full_path = lfs.currentdir()
path = ";"..string.sub(full_path, 1, -5).."baseline/baselines/ppo1/environment/?.lua"
-- print(path)
package.path=package.path .. path

require("common_functions")
require("ompl_functions")
require("robot_control")

-- simSetThreadSwitchTiming(2) 
-- simExtRemoteApiStart(19999)

_obs_mode = 'random' --'near'
_bound_x = 1.5
_bound_y = 1.5

_init_target_dist = 1
_target_dist = _init_target_dist
_max_obs_dist = 0.5

_new_ep_prob = 0
_modifly_prob = 0

function start()
    -- sleep (3)
    -- print('reset')
    _head_check_hd = simGetObjectHandle('head_check')
    _fake_robot_hd = simGetObjectHandle('fake_robot')
    _robot_hd = simGetObjectHandle('centauro')
    _robot_body_hd = simGetObjectHandle('body_ref')
    _target_hd = simGetObjectHandle('target')
    _joint_hds = get_joint_hds(16)

    _start_pos = simGetObjectPosition(_robot_hd, -1)
    _start_ori = simGetObjectOrientation(_robot_hd,-1)
    _start_joint_values = get_joint_values(_joint_hds)

    _start_t_pos = simGetObjectPosition(_target_hd, -1)
    _start_t_pos[3] = _start_t_pos[3] + 0.35
    _start_t_ori = simGetObjectOrientation(_target_hd,-1)

    _collection_hd = simGetCollectionHandle('obstacle_all')
    _collection_robot_hd = simGetCollectionHandle('centauro_mesh')

    _obstacles_hds = simGetCollectionObjects(_collection_hd)

    _obstacle_dynamic_collection = simGetCollectionHandle('obstacle_dynamic')
    _obstacle_dynamic_hds = simGetCollectionObjects(_obstacle_dynamic_collection)

    _obs_hds = check_avaiable_obstacle_hds()

    _current_ep = {}
    _pre_ep = {}
    _start_ep = {}
    _init_ep = convert_current_ep()

    _failed_ep_index = 1
    _failed_ep_history = {}
    _max_history_length = 1000
    _min_history_length = 0 --_max_history_length/4
    _sampl_node = 'new'
    _save_ep = false
    _g_save_ep = 0

    _pre_robot_pos = _start_pos
    _pre_robot_ori = _start_ori
    _pre_target_pos = _start_t_pos
    _pre_target_ori = _start_t_ori
    _pre_target_l = get_current_l(_robot_hd)
    print(_pre_target_l)

    _center_x = 0
    _center_y = 0

    -- print (_start_pos[1], _start_pos[2])
end

function check_avaiable_obstacle_hds()
    hds = {}
    for i=1, #_obstacle_dynamic_hds, 1 do 
        local obs_pos_global = simGetObjectPosition(_obstacle_dynamic_hds[i], -1)
        
        local x = math.abs(obs_pos_global[1])
        local y = math.abs(obs_pos_global[2])

        if x < 2.5 and y < 2.5 then   
            hds[#hds + 1] = _obstacle_dynamic_hds[i]
        end
    end
    return hds
end

function get_minimum_obs_dist(inInts,inFloats,inStrings,inBuffer)
    local threshold = 0.1
    local res, data = simCheckDistance(_collection_robot_hd, _collection_hd, threshold)
    if data == nil then 
        dist = threshold
    else 
        dist = data[7]
    end
    -- print(dist)    
    return {}, {dist}, {}, ''
end

function reset(inInts,inFloats,inStrings,inBuffer)
    local radius = inFloats[1]
    local env_mode = inFloats[2] --   0: random environment   1: target environment 
    local reset_mode = inFloats[3] --   0: pure random ep        1: allow replay buffer    2: continue... only change target position   3: play by user  4: close to goal
    _g_save_ep = inFloats[4]
    -- print ('reset', env_mode, reset_mode)

    if reset_mode == 4 then 
        _target_dist = 0.0
    else 
        _target_dist = _init_target_dist
    end

    if env_mode == 0 then 
        sample_ep(radius, reset_mode)  -- small random env
    elseif env_mode == 1 and reset_mode ~= 3 then
        sample_ep_fixenv(radius, reset_mode) -- initial scene
    elseif reset_mode == 3 then
        sample_ep_userplay(env_mode, reset_mode)
    end
    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    _center_x = robot_pos[1]
    _center_y = robot_pos[2]         
    return {}, {}, {}, ''
end

function step(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    _pre_ep = convert_current_ep()
    res = do_action_rl(_robot_hd, inFloats)
    _current_ep = convert_current_ep()
    return {}, {}, {}, res
end

function clear_history(inInts,inFloats,inStrings,inBuffer)
    _failed_ep_index = 1
    _failed_ep_history = {}
    return {}, {}, {}, ''
end

function clear_history_leave_one(inInts,inFloats,inStrings,inBuffer)
    if #_failed_ep_history > 0 then 
        _failed_ep_index = 2
        local ep = copy_table( _failed_ep_history[1])
        _failed_ep_history = {}
        _failed_ep_history[1] = ep
    end
    return {}, {}, {}, ''
end


function save_start_end_ep(inInts,inFloats,inStrings,inBuffer)
    if _save_ep and _g_save_ep == 1 then 
        _failed_ep_history[_failed_ep_index] = _start_ep
        _failed_ep_index = _failed_ep_index + 1
        _failed_ep_index = _failed_ep_index % _max_history_length

        local last_ep = convert_current_ep()
        _failed_ep_history[_failed_ep_index] = last_ep
        _failed_ep_index = _failed_ep_index + 1
        _failed_ep_index = _failed_ep_index % _max_history_length
        -- -- -- print('failed ep:', _failed_ep_index, #_failed_ep_history)
        print ('    save start ep', _failed_ep_index, #_failed_ep_history)
    end

    return {}, {}, {}, ''
end

function save_ep(inInts,inFloats,inStrings,inBuffer)
    local current_ep = convert_current_ep()
    _failed_ep_history[_failed_ep_index] = current_ep
    _failed_ep_index = _failed_ep_index + 1
    _failed_ep_index = _failed_ep_index % _max_history_length
    -- print('failed ep:', _failed_ep_index, #_failed_ep_history)
    print ('    save end ep')

    return {}, {}, {}, ''
end

function move_robot(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    robot_pos[1] =  inFloats[1]
    robot_pos[2] =  inFloats[2]

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    return {}, {}, {}, ''
end

function get_obstacle_info(inInts,inFloats,inStrings,inBuffer)

    local obs_info = {}
    for i=1, #_obs_hds, 1 do 
        -- local pos = simGetObjectPosition(_obs_hds[i], -1)
        local pos = simGetObjectPosition(_obs_hds[i], -1)
        local robot_pos =simGetObjectPosition(_robot_hd,-1)
        local res, type, dim = simGetShapeGeomInfo(_obs_hds[i])
        obs_info[#obs_info+1] = pos[1] - robot_pos[1]
        obs_info[#obs_info+1] = pos[2] - robot_pos[2]

        obs_info[#obs_info+1] = dim[1]
        obs_info[#obs_info+1] = dim[2]
        obs_info[#obs_info+1] = dim[3]

        -- print('shape: ', dim[1], dim[2], dim[3], dim[4])
    end

    return {}, obs_info, {}, ''
end

function get_robot_state(inInts,inFloats,inStrings,inBuffer)
    local target_pos =simGetObjectPosition(_target_hd, _robot_hd)
    local target_ori =simGetObjectPosition(_target_hd, _robot_hd)

    local pos =simGetObjectPosition(_robot_hd,-1)
    local ori =simGetObjectOrientation(_robot_hd,-1)
    -- local joint_pose = get_joint_values(_joint_hds)
    local leg_l = get_current_l(_robot_hd)
    local diff_x = math.abs(_center_x - pos[1])
    local diff_y = math.abs(_center_y - pos[2])
    local dist_to_center = math.sqrt(pos[1]*pos[1] + pos[2]*pos[2])
    -- local dist_to_center = math.sqrt(diff_x*diff_x + diff_y*diff_y)

    local state = {}
    local target_angle = math.atan2(target_pos[1], target_pos[2])
    local target_dist = math.sqrt(target_pos[1]*target_pos[1] + target_pos[2]*target_pos[2])

    state[1] = target_dist
    state[2] = dist_to_center
    state[3] = target_angle
    state[4] = target_dist  --(target_dist/(_target_dist*2)-0.5) * 2
    -- state[3] = target_ori[3]
    -- state[4] = target_pos[3] - 0.4
    -- state[5] = _pre_target_l

    local target_pos_head =simGetObjectPosition(_target_hd, _head_check_hd)
    state[#state+1] = target_pos_head[1]
    state[#state+1] = target_pos_head[2]
    state[#state+1] = target_pos_head[3]

    local min_dist = 999
    for i=1, #_obs_hds, 1 do 
        local obs_pos = simGetObjectPosition(_obs_hds[i], _head_check_hd)
        local obs_pos_g = simGetObjectPosition(_obs_hds[i], -1)
                        
        state[#state+1] = obs_pos[1]
        state[#state+1] = obs_pos[2]
        state[#state+1] = obs_pos[3]
        state[#state+1] = obs_pos_g[3]
    end
    state[#state+1] = ori[3]
    state[#state+1] = pos[3]
    state[#state+1] = leg_l

    local threshold = 0.3
    local res, data = simCheckDistance(_collection_robot_hd, _collection_hd, threshold)
    local dist = threshold
    if data ~= nil then 
        dist = data[7]
    end
    state[#state+1] = dist
    return {}, state, {}, ''
end

function generate_path()
    init_params(2, 8, 'centauro', 'obstacle_all', true)
    task_hd, state_dim = init_task('centauro','task_1')
    path = compute_path(task_hd, 10)
    print ('path found ', #path)
    -- displayInfo('finish 1 '..#path)

    for i=1, 30, 1 do 
        applyPath(task_hd, path, 0.1)
    end
    simExtOMPL_destroyTask(task_hd)

    return path
end

function applyPath(task_hd, path, speed)
    -- simSetModelProperty(robot_hd, 32)

    local state = {}
    for i=1,#path-state_dim,state_dim do
        for j=1,state_dim,1 do
            state[j]=path[i+j-1]
        end
        do_action_hl(_robot_hd, state)
        -- res = simExtOMPL_writeState(task_hd, state) 
        -- pos = {}
        -- pos[1] = state[1]
        -- pos[2] = state[2]
        -- pos[3] = 0
        -- print (pos[1])
        -- simSetObjectPosition(robot_hd, -1, pos)
        -- sleep (0.005)
        sleep(speed)
        simSwitchThread()
    end
    -- simSetModelProperty(robot_hd, 0)
end


-- function sample_obstacle_position()
--     for i=1, #_obs_hds, 1 do
--         local obs_pos = simGetObjectPosition(_obs_hds[i], -1)

--         if _obs_mode == 'random' then      
--             obs_pos[1] = (math.random()-0.5)*2 * _bound_x 
--             obs_pos[2] = (math.random()-0.5)*2 * _bound_y        
--         else 
--             obs_pos[1] = (math.random()-0.5)*2 * 0.05 + obs_pos[1]--(math.random()-0.5)*2 * _bound_x 
--             obs_pos[2] = (math.random()-0.5)*2 * 0.05 + obs_pos[2]--(math.random()-0.5)*2 * _bound_y 
--         end

--         if obs_pos[1] > _bound_x then
--             obs_pos[1] = _bound_x
--         elseif obs_pos[1] < -_bound_x then 
--             obs_pos[1] = -_bound_x
--         end

--         if obs_pos[2] > _bound_y then
--             obs_pos[2] = _bound_y
--         elseif obs_pos[2] < -_bound_y then 
--             obs_pos[2] = -_bound_y
--         end

--         simSetObjectPosition(_obs_hds[i], -1, obs_pos)
--     end
-- end

function sample_obstacle_position()
    local visable_count = 0
    local max_count = 10 --math.random(10)
    for i=1, #_obs_hds, 1 do
        local visable = math.random()
        if visable > 0.4 and visable_count < max_count then 
            local obs_pos = simGetObjectPosition(_obs_hds[i], -1)

            if _obs_mode == 'random' then      
                obs_pos[1] = (math.random()-0.5)*2 * 1
                obs_pos[2] = (math.random()-0.5)*2 * 1       
            else 
                obs_pos[1] = (math.random()-0.5)*2 * 0.05 + obs_pos[1]--(math.random()-0.5)*2 * _bound_x 
                obs_pos[2] = (math.random()-0.5)*2 * 0.05 + obs_pos[2]--(math.random()-0.5)*2 * _bound_y 
            end

            if obs_pos[1] > _bound_x then
                obs_pos[1] = _bound_x
            elseif obs_pos[1] < -_bound_x then 
                obs_pos[1] = -_bound_x
            end

            if obs_pos[2] > _bound_y then
                obs_pos[2] = _bound_y
            elseif obs_pos[2] < -_bound_y then 
                obs_pos[2] = -_bound_y
            end

            simSetObjectPosition(_obs_hds[i], -1, obs_pos)
            visable_count = visable_count + 1
        else 
            local obs_pos = simGetObjectPosition(_obs_hds[i], -1)
            obs_pos[1] = _bound_x*5
            obs_pos[2] = _bound_y*5  
            simSetObjectPosition(_obs_hds[i], -1, obs_pos)
        end 
    end
end

function sample_new_ep()

    sample_obstacle_position()

    local robot_pos = {}
    robot_pos[1] = 0 --(math.random() - 0.5) * 2 * 0.5
    robot_pos[2] = 0 --(math.random() - 0.5) * 2 * 0.5
    robot_pos[3] = _start_pos[3]

    local robot_ori = {}
    robot_ori[1] = _start_ori[1] 
    robot_ori[2] = _start_ori[2]
    robot_ori[3] = (math.random() - 0.5) *2 * math.pi

    local target_pos = {}
    target_pos[1] = 0 --(math.random() - 0.5) *2 + robot_pos[1] --* 2 * 0.5
    target_pos[2] = math.random() * _target_dist --* global_counter/20000
    target_pos[3] = _start_t_pos[3] --(math.random() - 0.5) * 2 * 0.1 + 0.4

    local target_ori = {}
    target_ori[1] = _start_t_ori[1] 
    target_ori[2] = _start_t_ori[2]
    target_ori[3] = (math.random() - 0.5) * 2 * math.pi

    simSetObjectPosition(_target_hd,-1,target_pos)
    simSetObjectOrientation(_target_hd, -1, target_ori)

    _pre_robot_pos = robot_pos
    _pre_robot_ori = robot_ori
    _pre_target_pos = target_pos
    _pre_target_ori = target_ori
    _pre_target_l = (math.random() - 0.5) * 2 * 0.05 + 0.07

    -- ep type
    if math.random() < 0.8 then 
        local obs_pos = {}
        local obs_index = math.random(#_obs_hds)
        local obs_pos_before =  simGetObjectPosition(_obs_hds[obs_index], -1)
        obs_pos[1] = (math.random() - 0.5)*2 * 0.4 + (robot_pos[1] + target_pos[1])/2
        obs_pos[2] = (math.random() - 0.5)*2 * 0.4 + (robot_pos[2] + target_pos[2])/2
        obs_pos[3] = obs_pos_before[3]
        simSetObjectPosition(_obs_hds[obs_index], -1, obs_pos)
    end

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    simSetObjectOrientation(_robot_hd, -1, robot_ori)
    set_joint_values(_joint_hds, _start_joint_values)

    -- print (res_robot, res_target)
    return check_collision(robot_pos, robot_ori, target_pos, target_ori)

end

function check_collision(robot_pos, robot_ori, target_pos, target_ori)
    -- return 0, 0
    -- simSetObjectPosition(_fake_robot_hd,-1,robot_pos)
    -- simSetObjectOrientation(_fake_robot_hd, -1, robot_ori)
    local res_robot = simCheckCollision(_collection_robot_hd, _collection_hd)

    simSetObjectPosition(_fake_robot_hd,-1,target_pos)
    simSetObjectOrientation(_fake_robot_hd, -1, target_ori)
    local res_target = simCheckCollision(_fake_robot_hd, _collection_hd)
    return res_robot, res_target
end

function sample_test_ep(start_ep, random_robot_pose)
    -- random_robot_pose = false
    restore_ep(start_ep, 0)   
    local target_x = 1
    local target_y = 1.2

    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    local robot_ori = simGetObjectOrientation(_robot_hd, -1)

    if random_robot_pose then 
        robot_pos[1] = (math.random() - 0.5) * 2 * _bound_x
        robot_pos[2] = (math.random() - 0.5) * 2 * _bound_y
        robot_pos[3] = _start_pos[3]

        robot_ori[1] = _start_ori[1] 
        robot_ori[2] = _start_ori[2]
        robot_ori[3] = (math.random() - 0.5) *2 * math.pi
    end 
    local target_pos = {}
    target_pos[1] = (math.random() - 0.5) *2 * _target_dist + robot_pos[1] 
    target_pos[2] = (math.random() - 0.5) *2 * _target_dist + robot_pos[2] 
    target_pos[3] = _start_t_pos[3] --(math.random() - 0.5) * 2 * 0.1 + 0.4

    if target_pos[1] > target_x then
        target_pos[1] = target_x
    elseif target_pos[1] < -target_x then 
        target_pos[1] = -target_x
    end

    if target_pos[2] > target_y then
        target_pos[2] = target_y
    elseif target_pos[2] < -target_y then 
        target_pos[2] = -target_y
    end

    local target_ori = {}
    target_ori[1] = _start_t_ori[1] 
    target_ori[2] = _start_t_ori[2]
    target_ori[3] = (math.random() - 0.5) * 2 * math.pi

    simSetObjectPosition(_target_hd,-1,target_pos)
    simSetObjectOrientation(_target_hd, -1, target_ori)

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    simSetObjectOrientation(_robot_hd, -1, robot_ori)
    set_joint_values(_joint_hds, _start_joint_values)

    _center_x = robot_pos[1]
    _center_y = robot_pos[2]

    -- print (res_robot, res_target)
    if random_robot_pose then 
        return check_collision(robot_pos, robot_ori, target_pos, target_ori)
    else 
        simSetObjectPosition(_fake_robot_hd,-1,target_pos)
        simSetObjectOrientation(_fake_robot_hd, -1, target_ori)
        local res_target = simCheckCollision(_fake_robot_hd, _collection_hd)
        local res_robot = 0
        return res_robot, res_target
    end
end

function convert_current_ep()
    local current_ep = {}
    local hds = {}
    local params = {}

    local obs_params = {}
    local robot_params = {}
    local target_params = {}
    -- hds 
    hds = {_obs_hds}
    hds[#hds+1] = _robot_hd
    hds[#hds+1] = _target_hd    

    -- obs_pos
    for i=1, #_obs_hds, 1 do
        local obs_pos = simGetObjectPosition(_obs_hds[i], -1)
        local obs_ori = simGetObjectOrientation(_obs_hds[i], -1)
        obs_params[#obs_params+1] = obs_pos 
        obs_params[#obs_params+1] = obs_ori 
    end

    -- robot pos, robot ori
    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    local robot_ori = simGetObjectOrientation(_robot_hd, -1)
    local robot_joints = get_joint_values(_joint_hds)
    robot_params[#robot_params+1] = robot_pos 
    robot_params[#robot_params+1] = robot_ori 
    robot_params[#robot_params+1] = robot_joints 

    -- target pos, target ori 
    local target_pos = simGetObjectPosition(_target_hd, -1)
    local target_ori = simGetObjectOrientation(_target_hd, -1)
    target_params[#target_params+1] = target_pos 
    target_params[#target_params+1] = target_ori 

    params[1] = obs_params
    params[2] = robot_params
    params[3] = target_params        

    current_ep[1] = hds
    current_ep[2] = params

    return current_ep
end 

function restore_ep(ep, modifly)
    local episode = copy_table(ep) 

    local shift = 0.2
    local robot_shift = 0.1
    local hds = episode[1]
    local params = episode[2]

    local obs_hds = hds[1]
    local robot_hd = hds[2]
    local target_hd = hds[3]

    local obs_params = params[1]
    local robot_params = params[2]
    local target_params = params[3]
    
    -- obs
    for i=1, #obs_hds, 1 do
        local param_index = i*2 - 1
        local obs_pos = obs_params[param_index]
        local obs_ori = obs_params[param_index+1]

        -- if modifly > 0.5 then 
        --     obs_pos[1] = obs_pos[1] + (math.random() - 0.5) *2 * shift
        --     obs_pos[2] = obs_pos[2] + (math.random() - 0.5) *2 * shift
        -- end 

        if obs_pos[1] > _bound_x and  obs_pos[1] < 2.5 then
            obs_pos[1] = _bound_x
        elseif obs_pos[1] < -_bound_x  and  obs_pos[1] < 2.5 then 
            obs_pos[1] = -_bound_x
        end

        if obs_pos[2] > _bound_y  and  obs_pos[2] < 2.5 then
            obs_pos[2] = _bound_y
        elseif obs_pos[2] < -_bound_y  and  obs_pos[2] < 2.5 then 
            obs_pos[2] = -_bound_y
        end
        simSetObjectPosition(obs_hds[i], -1, obs_pos)
        simSetObjectOrientation(obs_hds[i], -1, obs_ori)
    end

    local robot_pos = robot_params[1]
    local robot_ori = robot_params[2]
    local robot_joints =  robot_params[3]

    local target_pos = target_params[1]
    local target_ori = target_params[2]

    -- robot 
    if modifly > 0.5 or _target_dist ~= _init_target_dist then 
        -- local x_mid = (target_pos[1] + robot_pos[1] )/2
        -- local y_mid = (target_pos[2] + robot_pos[2] )/2    
        -- robot_pos[1] = (math.random() - 0.5) *2 * _target_dist/2 + x_mid
        -- robot_pos[2] = (math.random() - 0.5) *2 * _target_dist/2 + y_mid
        local modifly_tpye =  math.random(3)
        if _target_dist ~= _init_target_dist then 
            shift = 0.0
            modifly_tpye = 3
        end        
        if modifly_tpye == 1 then 
            local x_mid = (target_pos[1] + robot_pos[1] )/2
            local y_mid = (target_pos[2] + robot_pos[2] )/2    
            robot_pos[1] = (math.random() - 0.5) *2 * _target_dist/2 + x_mid
            robot_pos[2] = (math.random() - 0.5) *2 * _target_dist/2 + y_mid
        elseif modifly_tpye == 2 then    
            robot_pos[1] = (math.random() - 0.5) *2 * shift + robot_pos[1]
            robot_pos[2] = (math.random() - 0.5) *2 * shift + robot_pos[2]
        else 
            robot_pos[1] = (math.random() - 0.5) *2 * shift + target_pos[1]
            robot_pos[2] = (math.random() - 0.5) *2 * shift + target_pos[2]
        end 
        robot_ori[3] = (math.random() - 0.5) *2 * math.pi
    end 
    simSetObjectPosition(robot_hd, -1, robot_pos)
    simSetObjectOrientation(robot_hd, -1, robot_ori)
    set_joint_values(_joint_hds, robot_joints)

    -- -- target 
    -- if modifly > 0.5 then 
    --     target_pos[1] = target_pos[1] + (math.random() - 0.5) *2 * shift    
    --     target_pos[2] = target_pos[2] + (math.random() - 0.5) *2 * shift       
    -- end 

    simSetObjectPosition(target_hd, -1, target_pos)
    simSetObjectOrientation(target_hd, -1, target_ori)

    if modifly > 0.5 then 
        return check_collision(robot_pos, robot_ori, target_pos, target_ori)
    else 
        return 0, 0
    end
end



function sample_ep_fixenv(radius, reset_mode)
    global_counter = global_counter + 1

    local restore_or_new = math.random()
    local modifly = math.random()

    if reset_mode == 2 then-- continue in the same environment, only change the target position
        _sampl_node = ' continue'
        local ep = convert_current_ep()
        local robot_res, target_res = sample_test_ep(ep, false)      
        while robot_res==1 or target_res==1 do
            robot_res, target_res = sample_test_ep(ep, false)      
        end       
    else -- use buffer or random sample robot and target position
        if restore_or_new > 0.3 and #_failed_ep_history > 1 and reset_mode == 1 then       -- restore failure ep
            _sampl_node = 'replay'
            _save_ep = false

            local sample_failed_ep_index = math.random(#_failed_ep_history)
            local ep = _failed_ep_history[sample_failed_ep_index]

            if modifly > 0.5 then 
                _sampl_node = 'replay_modifly'
                _save_ep = false
            end
            local robot_res = 1
            local target_res = 1
            local count = 0
            while robot_res==1 or target_res==1 do
                robot_res, target_res = restore_ep(ep, modifly)
                count = count + 1
                if count > 15 then 
                    _sampl_node = 'replay'
                    modifly = 0
                end
            end
        else -- sample a random one in target environment
            local random_robot_pose = true
            _sampl_node = 'random'
            _save_ep = true
            random_robot_pose = true

            local robot_res, target_res = sample_test_ep(_init_ep, random_robot_pose)
            while robot_res==1 or target_res==1 do
                robot_res, target_res = sample_test_ep(_init_ep, random_robot_pose)          
            end                
        end
    end

    -- _save_ep = false
    _start_ep = convert_current_ep()
    print('fix env', _sampl_node, _save_ep, _failed_ep_index, #_failed_ep_history)
end

function sample_ep(radius, reset_mode)  -- sample small training case: robot in the middle, target locate between 0 to 1
    global_counter = global_counter + 1

    local sample_type = math.random()
    if reset_mode == 3 then
        restore_ep(_current_ep, 0) 
    elseif sample_type > _new_ep_prob and #_failed_ep_history > _min_history_length and (reset_mode == 1 or reset_mode == 4) then       -- restore failure ep
        _sampl_node = 'replay'

        local sample_failed_ep_index = math.random(#_failed_ep_history)
        local ep = _failed_ep_history[sample_failed_ep_index]

        local modifly = math.random()
        if modifly > (1-_modifly_prob) then 
            _sampl_node = 'replay_modifly'
            modifly = 1
        else 
            modifly = 0
        end
        local robot_res = 1
        local target_res = 1
        local count = 1
        while robot_res==1 or target_res==1 do
            robot_res, target_res = restore_ep(ep, modifly) 
            count = count + 1
            if count > 15 then 
                _sampl_node = 'replay'
                modifly = 0
            end
        end

    else    -- generate random ep
        _sampl_node = 'new'

        local robot_res = 1
        local target_res = 1
        while robot_res==1 or target_res==1 do
            robot_res, target_res = sample_new_ep()
        end
    end

    if _sampl_node == 'new' or _sampl_node == 'replay_modifly' then 
        _save_ep = true 
    else 
        _save_ep = false 
    end 
    -- _save_ep = true 

    _start_ep = convert_current_ep() 
    print('training', _sampl_node, _save_ep, _failed_ep_index, #_failed_ep_history)
end


function sample_ep_userplay(env_mode, reset_mode)  -- fixed environment, continue robot and target position
    print (env_mode, reset_mode)
    if env_mode == 1 then -- target env
        if reset_mode < 0 then --init env
            random_robot_pose = false
            local robot_res, target_res = sample_test_ep(_init_ep, random_robot_pose)
        end
    else    -- random env
        if reset_mode < 0 then --init env
            local robot_res = 1
            local target_res = 1
            while robot_res==1 or target_res==1 do
                robot_res, target_res = sample_new_ep()
            end
        end
    end

    _start_ep = convert_current_ep() 
    print('usr play', _sampl_node, _save_ep, _failed_ep_index, #_failed_ep_history)
end


initialized = false
global_counter = 0

start()
_current_ep = convert_current_ep() 
-- generate_path()
-- sample_ep(1.5)

-- for i=1, #_joint_hds, 1 do
-- local hd = _joint_hds[5]
-- local hd=simGetObjectHandle('j_wheel_2')
-- simSetJointTargetVelocity(hd, -100)
-- simSetJointForce(hd, 1)
-- simSwitchThread()
-- end 


while simGetSimulationState()~=sim_simulation_advancing_abouttostop do
    -- do something in here
    -- simSwitchThread()
end




