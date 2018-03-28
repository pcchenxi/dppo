local lfs = require("lfs")
full_path = lfs.currentdir()
path = ";"..string.sub(full_path, 1, -5).."baseline/baselines/ppo1/environment/?.lua"
--print(path)
package.path=package.path .. path
require("get_set")

-- action: vx, vy, vw, vl

scale = 1
step = 0.05 * scale
dx = step * scale
dy = step * scale
dh = 0.02 * scale
dl = 0.02 * scale
dw = math.pi/180 * 5 * scale

collision_hd_1 = simGetCollectionHandle('centauro_mesh')
collision_hd_2 = simGetCollectionHandle('obstacle_all')

control_joint_hds = get_joint_hds(8)
control_joint_hd_all = get_joint_hds(12)

function look_at_target()
    local t_pos = simGetObjectPosition(_target_hd, _head_check_hd)
    local hc_ori = simGetObjectOrientation(_head_check_hd,-1)
    
    local yaw = -math.atan2(t_pos[1], t_pos[2])
    hc_ori[3] = yaw + hc_ori[3]
    simSetObjectOrientation(_head_check_hd, -1, hc_ori)
end 

function is_valid()
    local valid = true
    local reasom = ''
    local res=simCheckCollision(collision_hd_1, collision_hd_2)
    reasom = 'pass'
    if res > 0 then 
        valid = false
        reasom = 'collision'
    end

    for i=1, #control_joint_hd_all, 1 do
        local joint_position = simGetObjectPosition(control_joint_hd_all[i], -1)
        if joint_position[3] < 0.10 then
            valid = false
            reasom = 'body down '..i
            break
        end
    end
    -- print(reasom, valid)
    return valid
end

function do_action_rl(robot_hd, action)
    -- print ('action_rl', action[1], action[2], action[3], action[4], action[5])

    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectOrientation(robot_hd,-1)
    local current_joint_values = get_joint_values(_joint_hds)
    local leg_l = get_current_l(robot_hd)


    -- local diff_l = 1 - math.abs(leg_l - 0.13)/0.2
    -- local diff_h = 1 - math.abs(current_pos[3])/0.2
    -- local ratio = diff_l
    -- if diff_h < diff_l then 
    --     ratio = diff_h
    -- end 
    -- action[1] = action[1] * ratio
    -- action[2] = action[2] * ratio

    local sample_pose = {}
    sample_pose[1] = dx*action[1]
    sample_pose[2] = dy*action[2] 
    sample_pose[3] = dh*action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = current_ori[3] + dw*action[3] 

    if sample_ori[3] > math.pi then
        sample_ori[3] = sample_ori[3]-math.pi*2
    end
    if sample_ori[3] < -math.pi then
        sample_ori[3] = sample_ori[3]+math.pi*2
    end

    simSetObjectPosition(robot_hd,robot_hd,sample_pose)
    simSetObjectOrientation(robot_hd,-1,sample_ori)

    local new_pos=simGetObjectPosition(robot_hd,-1)
    if math.abs(new_pos[1]) > 20 then 
        result = 'c'
    else 
        action[5] = leg_l + dl*action[5]
        -- print(leg_l, action[5])
        result = do_action(robot_hd, action)

        -- local new_pos=simGetObjectPosition(robot_hd,-1)
        -- if math.abs(new_pos[1])>0.5 or math.abs(new_pos[2])>0.5 then 
        --     result = 'a'
        -- end
    end 

    if result ~= 't' then 
    -- if result == 'a' then
        simSetObjectPosition(robot_hd,-1,current_pos)
        simSetObjectOrientation(robot_hd,-1,current_ori)
        set_joint_values(_joint_hds, current_joint_values)
    end

    return result
    -- return sample_pose, sample_ori
end

function do_action_hl(robot_hd, action)
    -- print(#action)           
    -- print (action[1], action[2], action[3], action[4], action[5])
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectOrientation(robot_hd,-1)

    local sample_pose = {}
    sample_pose[1] = action[1]
    sample_pose[2] = action[2] 
    sample_pose[3] = action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = action[3] 

    if sample_ori[3] > math.pi then
        sample_ori[3] = sample_ori[3]-math.pi*2
    end
    if sample_ori[3] < -math.pi then
        sample_ori[3] = sample_ori[3]+math.pi*2
    end

    simSetObjectPosition(robot_hd, -1, sample_pose)
    simSetObjectOrientation(robot_hd, -1, sample_ori)

    res = do_action(robot_hd, action)
    return res
end

function do_action(robot_hd, action)
    -- print ('action ', action[1], action[2], action[3], action[4], action[5])

    -- action is the final value to send
    local tilt_pos = {}
    local foot_pos = {}

    -- sample feet and joint --
    local r0 = 0.15 --math.sqrt(knee_pose[1]^2 + knee_pose[2]^2)
    local r1 = 0.1 --math.sqrt(ankle_pose[1]^2 + ankle_pose[2]^2)    

    -- for i=1, 4, 1 do
    local dummy_ankle_hd = simGetObjectHandle('temp_ankle')
    local dummy_wheel_hd = simGetObjectHandle('temp_wheel')
    local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    local hip_hd = simGetObjectHandle('hip_pitch_1')
    local hip_ori_hd = simGetObjectHandle('hip_ori_ref')
    local knee_hd = simGetObjectHandle('knee_pitch_1')

    -- set dummy for target ankle position
    hip_pos = simGetObjectPosition(hip_hd, -1)
    local ankle_target = {}
    ankle_target[1] = 0.17 - hip_pos[3]
    ankle_target[2] = action[5]
    ankle_target[3] = 0
    simSetObjectPosition(dummy_ankle_hd, hip_ori_hd, ankle_target)
    -- print('ankle_target:', i, ankle_target[1], ankle_target[2], ankle_target[3])

    -- set dummy for target wheel position
    local wheel_target = ankle_target
    wheel_target[1] = wheel_target[1] - 0.1175
    simSetObjectPosition(dummy_wheel_hd, hip_ori_hd, wheel_target)


    local ankle_in_hip = simGetObjectPosition(dummy_ankle_hd, hip_ori_hd)
    -- print('ankle_target:', i, ankle_in_hip[1], ankle_in_hip[2], ankle_in_hip[3])

    local knee_x, knee_y = get_intersection_point(0, 0, ankle_in_hip[1], ankle_in_hip[2], r0, r1)
    -- simSetObjectPosition(dummy_wheel_hd, hip_hd, {knee_x, knee_y, ankle_in_hip[3]})
    -- print('knee_target:', i, knee_x, knee_y)

    if knee_x == -1 or knee_x ~= knee_x then
        return 'a'
    end

    local angle_hip = -math.atan2(knee_x, knee_y)
    local hip_p = simGetJointPosition(hip_hd)
    simSetJointPosition(hip_hd, angle_hip)

    local ankle_in_knee = simGetObjectPosition(dummy_ankle_hd, knee_hd)
    local angle_knee = math.atan2(ankle_in_knee[2], ankle_in_knee[1])
    local knee_p = simGetJointPosition(knee_hd)
    simSetJointPosition(knee_hd, angle_knee)

    local wheel_in_ankle = simGetObjectPosition(dummy_wheel_hd, ankle_hd)
    local angle_ankle = math.atan2(wheel_in_ankle[2], wheel_in_ankle[1])      
    simSetJointPosition(ankle_hd, angle_ankle)


    local ankle2_hd = simGetObjectHandle('ankle_pitch_2')
    local hip2_hd = simGetObjectHandle('hip_pitch_2')
    local knee2_hd = simGetObjectHandle('knee_pitch_2')
    simSetJointPosition(hip2_hd, -angle_hip)
    simSetJointPosition(knee2_hd, -angle_knee)
    simSetJointPosition(ankle2_hd, -angle_ankle)

    local ankle3_hd = simGetObjectHandle('ankle_pitch_3')
    local hip3_hd = simGetObjectHandle('hip_pitch_3')
    local knee3_hd = simGetObjectHandle('knee_pitch_3')
    simSetJointPosition(hip3_hd, -angle_hip)
    simSetJointPosition(knee3_hd, -angle_knee)
    simSetJointPosition(ankle3_hd, -angle_ankle)

    local ankle4_hd = simGetObjectHandle('ankle_pitch_4')
    local hip4_hd = simGetObjectHandle('hip_pitch_4')
    local knee4_hd = simGetObjectHandle('knee_pitch_4')
    simSetJointPosition(hip4_hd, angle_hip)
    simSetJointPosition(knee4_hd, angle_knee)
    simSetJointPosition(ankle4_hd, angle_ankle)

    look_at_target()
    -- check collision --
    if is_valid() then 
        return 't'
    else
        -- displayInfo('collide '..i..' '..foot_pos[1]..' '..foot_pos[2] )
        return 'c'      
    end
end

get_intersection_point=function(x0, y0, x1, y1, r0, r1)
    local d=math.sqrt((x1-x0)^2 + (y1-y0)^2)
    if d>(r0+r1) then
        return -1, -1
    end
    
    local a=(r0^2-r1^2+d^2)/(2*d)
    local h=math.sqrt(r0^2-a^2)
    local x2=x0+a*(x1-x0)/d   
    local y2=y0+a*(y1-y0)/d   
    local x3_1=x2+h*(y1-y0)/d       -- also x3=x2-h*(y1-y0)/d
    local y3_1=y2-h*(x1-x0)/d       -- also y3=y2+h*(x1-x0)/d

    local x3_2=x2-h*(y1-y0)/d       -- also x3=x2-h*(y1-y0)/d
    local y3_2=y2+h*(x1-x0)/d       -- also y3=y2+h*(x1-x0)/d

    if y3_1 > y3_2 then
        return x3_1, y3_1
    else 
        return x3_2, y3_2
    end
end

function get_current_l(robot_hd)
    -- local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    -- local hip_hd = simGetObjectHandle('hip_ori_ref')

    -- local foot_pos = simGetObjectPosition(ankle_hd, hip_hd)

    -- return math.abs(foot_pos[2])
    return 0.1
end
