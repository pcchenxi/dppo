function get_joint_hds(joint_num)
   
    local hd1_1=simGetObjectHandle('hip_yaw_1')
    local hd1_2=simGetObjectHandle('hip_yaw_2')
    local hd1_3=simGetObjectHandle('hip_yaw_3')
    local hd1_4=simGetObjectHandle('hip_yaw_4')

    local hd2_1=simGetObjectHandle('hip_pitch_1')
    local hd2_2=simGetObjectHandle('hip_pitch_2')
    local hd2_3=simGetObjectHandle('hip_pitch_3')
    local hd2_4=simGetObjectHandle('hip_pitch_4')

    local hd3_1=simGetObjectHandle('knee_pitch_1')
    local hd3_2=simGetObjectHandle('knee_pitch_2')
    local hd3_3=simGetObjectHandle('knee_pitch_3')
    local hd3_4=simGetObjectHandle('knee_pitch_4')

    local hd4_1=simGetObjectHandle('ankle_pitch_1')
    local hd4_2=simGetObjectHandle('ankle_pitch_2')
    local hd4_3=simGetObjectHandle('ankle_pitch_3')
    local hd4_4=simGetObjectHandle('ankle_pitch_4')

    local hd5_1=simGetObjectHandle('j_wheel_1')
    local hd5_2=simGetObjectHandle('j_wheel_2')
    local hd5_3=simGetObjectHandle('j_wheel_3')
    local hd5_4=simGetObjectHandle('j_wheel_4')

    joint_hds = {}
    if joint_num == 8 then
        joint_hds = {hd2_1,hd2_2,hd2_3,hd2_4, hd3_1,hd3_2,hd3_3,hd3_4}
    elseif joint_num == 12 then
        joint_hds = {hd2_1,hd2_2,hd2_3,hd2_4, hd3_1,hd3_2,hd3_3,hd3_4, hd4_1,hd4_2,hd4_3,hd4_4}
    elseif joint_num == 16 then 
        joint_hds={hd1_1,hd1_2,hd1_3,hd1_4, hd2_1,hd2_2,hd2_3,hd2_4, hd3_1,hd3_2,hd3_3,hd3_4, hd4_1,hd4_2,hd4_3,hd4_4}

        -- joint_hds={hd1_1,hd1_2,hd1_3,hd1_4, hd2_1,hd2_2,hd2_3,hd2_4, hd3_1,hd3_2,hd3_3,hd3_4, hd4_1,hd4_2,hd4_3,hd4_4, hd5_1,hd5_2,hd5_3,hd5_4}
    end
    return joint_hds
end

function get_joint_values(joint_hds)
    local joint_values={}
    for i=1, #joint_hds, 1 do
        joint_values[i] = simGetJointPosition(joint_hds[i])
    end
    return joint_values
end

set_joint_values = function(joint_hds, joint_values)
    for i=1, #joint_hds, 1 do
        res = simSetJointPosition(joint_hds[i], joint_values[i])
    end
end

function get_state(hd, joint_hds, base_dim, joint_dim)
    local pos =simGetObjectPosition(hd,-1)
    local ori =simGetObjectQuaternion(hd,-1)
    local joint_pose = get_joint_values(joint_hds)

    local state = {}
    if base_dim == 1 then         -- xy
        state[1] = pos[1]
        state[2] = pos[2]
    elseif base_dim == 2 then    -- xy yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = ori[3]
    elseif base_dim == 3 then    -- xyz yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = pos[3]
        state[4] = ori[3]
    elseif base_dim == 4 then    -- xyz roll pitch yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = pos[3]
        state[4] = ori[1]
        state[5] = ori[2]
        state[6] = ori[3]
    end

    for i=1, joint_dim, 1 do
        state[#state+1] = joint_pose[i]
    end

    return state
end

function get_state_hl(hd, base_dim)
    local pos =simGetObjectPosition(hd,-1)
    local ori =simGetObjectQuaternion(hd,-1)

    local state = {}
    if base_dim == 1 then         -- xy
        state[1] = pos[1]
        state[2] = pos[2]
    elseif base_dim == 2 then    -- xy yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = ori[3]
    elseif base_dim == 3 then    -- xyz yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = pos[3]
        state[4] = ori[3]
    elseif base_dim == 4 then    -- xyz roll pitch yaw
        state[1] = pos[1]
        state[2] = pos[2]
        state[3] = pos[3]
        state[4] = ori[1]
        state[5] = ori[2]
        state[6] = ori[3]
    end

    local hip_hd = simGetObjectHandle('hip_pitch_1')
    local ankle_hd = simGetObjectHandle('ankle_pitch_1')
    local robot_hd = simGetObjectHandle('centauro')

    local hip_pos =simGetObjectPosition(hip_hd, robot_hd)
    local ankle_pos =simGetObjectPosition(ankle_hd, robot_hd)

    state[#state+1] = pos[3]
    state[#state+1] = ankle_pos[1] - hip_pos[1]

    return state
end