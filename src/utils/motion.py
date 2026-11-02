import torch
import numpy as np
from .rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)

def accumulate_rotations_batch(R_rel):
    B, T, _, _ = R_rel.shape

    R_list = []
    R_prev = R_rel[:, 0]
    R_list.append(R_prev)

    for t in range(1, T):
        R_curr = R_rel[:, t] @ R_prev
        R_list.append(R_curr)
        R_prev = R_curr

    R_out = torch.stack(R_list, dim=1)
    return R_out

def recover_from_local_position_fast(final_x: torch.Tensor, njoint=22):
    """
    final_x: (B, T, D)
    return: (B, T, J, 3)
    """
    B, T, _ = final_x.shape
    device = final_x.device

    # -------- unpack --------
    positions_no_heading = final_x[:, :, 8:8+3*njoint].reshape(B, T, njoint, 3)
    velocities_root_xy = final_x[:, :, :2]
    global_heading_diff_rot = final_x[:, :, 2:8]

    # -------- rotation --------
    R_rel = rotation_6d_to_matrix(global_heading_diff_rot)  # (B, T, 3, 3)
    R_global = accumulate_rotations_batch(R_rel)            # (B, T, 3, 3)
    R_inv = R_global.transpose(-1, -2)                      # inverse

    # -------- rotate joints --------
    pos = torch.matmul(
        R_inv[:, :, None, :, :], 
        positions_no_heading[..., None]
    ).squeeze(-1)                                           # (B, T, J, 3)

    # -------- root velocity --------
    # Functional updates preserve autograd through the decoder latent during
    # trajectory fitting. In-place slice writes invalidate the backward graph.
    zero = torch.zeros_like(velocities_root_xy[:, :, 0])
    vel_root = torch.stack((velocities_root_xy[:, :, 0], zero,
                            velocities_root_xy[:, :, 1]), dim=-1)
    rotated_velocity = torch.matmul(
        R_inv[:, :-1],
        vel_root[:, 1:, :, None]
    ).squeeze(-1)
    vel_root = torch.cat((vel_root[:, :1], rotated_velocity), dim=1)

    # -------- integrate --------
    root_translation = torch.cumsum(vel_root, dim=1)

    # Preserve the original planar-integration convention even when a
    # predicted heading matrix has a small non-yaw component.
    horizontal_translation = torch.stack((root_translation[:, :, 0],
                                          torch.zeros_like(root_translation[:, :, 1]),
                                          root_translation[:, :, 2]), dim=-1)
    return pos + horizontal_translation[:, :, None, :]

def recover_from_local_position(final_x, njoint):

    if final_x.ndim == 3:
        bs, nfrm, _ = final_x.shape
        is_batched = True
    else:
        nfrm, _ = final_x.shape
        bs = 1
        is_batched = False
        final_x = final_x.reshape(1, *final_x.shape)

    
    positions_no_heading = final_x[:,:,8:8+3*njoint].reshape(bs, nfrm, njoint, 3) 
    velocities_root_xy_no_heading = final_x[:,:,:2] 
    global_heading_diff_rot = final_x[:,:,2:8] 
    
    positions_with_heading = []
    for b in range(bs):
        
        global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot[b])).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        
        curr_pos_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), 
                                        positions_no_heading[b][...,None]).squeeze(-1)
        
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading[b].shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[b, :, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[b, :, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], 
                                                         velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
        
        curr_pos_with_heading[:, :, 0] += root_translation[:, 0:1]
        curr_pos_with_heading[:, :, 2] += root_translation[:, 2:]
        
        positions_with_heading.append(curr_pos_with_heading)

    positions_with_heading = np.stack(positions_with_heading, axis=0)

    if not is_batched:
        positions_with_heading = positions_with_heading.squeeze(0)

    return positions_with_heading

def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])
    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height

    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    global_orient = axis_angle[:, :3]

    # y-up -> z-up
    # global_orient_mat = torch.from_numpy(rotations_matrix)[:, 0].float()
    # rotation_matrix = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).float()
    # global_orient_mat = torch.einsum('ij,tjk->tik', rotation_matrix, global_orient_mat)
    # global_orient = matrix_to_axis_angle(global_orient_mat)
    # root_translation = torch.einsum('ij,tj->ti', rotation_matrix, torch.from_numpy(root_translation).float())
    
    return dict(
        global_orient = global_orient, 
        body_pose = axis_angle[:, 3:], 
        transl = root_translation, 
        betas = torch.zeros((nfrm, 10))
    )
