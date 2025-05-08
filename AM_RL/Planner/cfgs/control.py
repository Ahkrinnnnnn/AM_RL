import torch

dev = 'cuda'

# PD gains
kp_pos = torch.tensor([0.7, 0.7, 0.7], device=dev)  # X, Y, Z
kd_pos = torch.tensor([0.1, 0.1, 0.1], device=dev)
kp_att = torch.tensor([0.20, 0.20, 0.20], device=dev)   # Roll, Pith, Yaw
kd_att = torch.tensor([0.10, 0.10, 0.10], device=dev)

def pd_control(current_pos, current_vel, current_ang_vel, target_pos, target_yaw):
    # Postion error (convert to body frame)
    error_pos = target_pos - current_pos
    error_vel = -current_vel  # Damping
                
    # Desred thrust (Z-axis in body frame)
    thrust = kp_pos[2] * error_pos[:, 2] + kd_pos[2] * error_vel[:, 2]
    thrust = torch.cat([
        torch.zeros(thrust.shape[0], 6, 2, device=thrust.device),
        (thrust / 6).view(-1, 1, 1).expand(thrust.shape[0], 6, 1)
    ], dim=2)

    # Desired roll/pitch from X/Y errors (simplified)
    desired_pitch = kp_pos[0] * error_pos[:, 0] + kd_pos[0] * error_vel[:, 0]
    desired_roll = kp_pos[1] * error_pos[:, 1] + kd_pos[1] * error_vel[:, 1]
                
    # Attitude control (convert to torques)
    # (In practice, use a quaternion/rotation matrix here)
    torques = torch.stack([
        kp_att[0] * desired_roll + kd_att[0] * (-current_ang_vel[:, 0]),  # Roll torque
        kp_att[1] * desired_pitch + kd_att[1] * (-current_ang_vel[:, 1]),  # Pitch torque
        kp_att[2] * target_yaw + kd_att[2] * (-current_ang_vel[:, 2])      # Yawtorque
    ], dim=1).unsqueeze(1)
    torques = (torques / 6).expand(torques.shape[0], 6, 3)

    return thrust.float(), torques.float()
