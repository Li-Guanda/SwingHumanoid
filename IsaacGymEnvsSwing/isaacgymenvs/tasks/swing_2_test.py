# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import random

import numpy as np
import os
import torch
import math
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import sys
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


class Swing2test(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = False
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 47
        self.cfg["env"]["numActions"] = 2
        self.cfg["env"]["numBodies"] = 21

        self.actions_cost_scale = 20
        self.energy_cost_scale = 20
        self.amplitude = 0.1
        self.frequency = 1.5
        self.stiffness_list = [125, 25, 125, 25]


        # self.output_dof_vel_tensor = torch.zeros((self.cfg["env"]["numActions"] * 2, self.num_envs, self.max_episode_length))

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


        self.target_buffer = torch.zeros((self.num_envs + 1, self.max_episode_length), device=self.device)

        for k in range(self.num_envs + 1):
            for i in range(self.max_episode_length):
                self.target_buffer[k, i] = -self.amplitude * math.cos(
                    i * 2 * torch.pi * 0.0166 * 0.1 * k * self.frequency + math.acos(1)) + self.amplitude

        self.output_dof_pos_tensor = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"] * 2, self.max_episode_length))
        self.output_dof_vel_tensor = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"] * 2, self.max_episode_length))
        self.output_actuator_force_tensor = torch.zeros(
            (self.num_envs, self.cfg["env"]["numActions"] * 2, self.max_episode_length))
        self.head_pos_tensor = torch.zeros((self.num_envs, self.max_episode_length))
        self.head_vel_tensor = torch.zeros((self.num_envs, self.max_episode_length))
        self.target_pos_tensor = torch.zeros((self.num_envs, self.max_episode_length))
        self.distance_tensor = torch.zeros((self.num_envs, self.max_episode_length))
        self.cop_tensor = torch.zeros((self.num_envs, self.max_episode_length))
        self.target_frequency = torch.range(1, self.num_envs, dtype=torch.long, device=self.device)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(5.0, 20.0, 5.0)
            cam_target = gymapi.Vec3(5.0, 5.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs,
                                                                          2 * self.cfg["env"]["numActions"],
                                                                          2)
        vec_body_tensor = gymtorch.wrap_tensor(self.body_state_tensor).view(self.num_envs, self.cfg["env"]["numBodies"],
                                                                            13)

        contact_force_tensor=self.gym.acquire_net_contact_force_tensor(self.sim)
        self.vec_contact_force_tensor=gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)
        self.initial_vec_root_tensor = vec_root_tensor.clone()
        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]

        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]

        self.dof_states = vec_dof_tensor
        self.dof_pos = vec_dof_tensor[..., 0]
        self.dof_vel = vec_dof_tensor[..., 1]

        self.body_states = vec_body_tensor
        self.body_pos_p = vec_body_tensor[..., 0:3]
        self.body_pos_r = vec_body_tensor[..., 3:7]
        self.body_vel_p = vec_body_tensor[..., 7:10]
        self.body_vel_r = vec_body_tensor[..., 10:13]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()
        self.initial_body_states = self.body_states.clone()
        self.initial_body_pos_p = self.body_pos_p.clone()

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 1.7

        self.time_step = 0

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.basis_vec1 = self.up_vec.clone()
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape(
            (self.num_envs, 2))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], 10)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_swing_2.xml"


        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        print(motor_efforts)
        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.29, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.075, asset_options)

        default_pose = gymapi.Transform()
        default_pose.p.z = 3

        self.actor_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, actor_handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            marker_handle = self.gym.create_actor(env_ptr, marker_asset, default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 0, 0))

            self.actor_handles.append(actor_handle)
            self.envs.append(env_ptr)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
            dof_props['stiffness'] = self.stiffness_list
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
        # dof_props_test = self.gym.get_actor_dof_properties(env_ptr, actor_handle)

        for j in range(self.num_dof):
            if dof_props['lower'][j] > dof_props['upper'][j]:
                self.dof_limits_lower.append(dof_props['upper'][j])
                self.dof_limits_upper.append(dof_props['lower'][j])
            else:
                self.dof_limits_lower.append(dof_props['lower'][j])
                self.dof_limits_upper.append(dof_props['upper'][j])
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def set_targets(self):

        self.target_root_positions[:, 0] = self.target_buffer[self.target_frequency[:], self.progress_buf[:]]
        # for i in range(self.max_episode_length):
        #     self.target_root_positions[i,0]=self.target_buffer[0,]
        self.marker_positions[:] = self.target_root_positions[:]

        # target_indices=self.all_actor_indices[:, 1].flatten()
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
        #                                          gymtorch.unwrap_tensor(target_indices), len(target_indices))

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
            self.marker_positions,
            self.body_states,
            self.initial_body_pos_p,
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.obs_buf[:], self.up_vec[:] = compute_humanoid_observations(
            self.body_states,
            self.root_states,
            self.inv_start_rot,
            self.dof_pos,
            self.dof_vel,
            self.dof_force_tensor,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.dof_vel_scale,
            self.vec_sensor_tensor,
            self.actions,
            self.contact_force_scale,
            self.angular_velocity_scale,
            self.basis_vec1,
            self.marker_positions,
        )

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        self.target_frequency = torch.range(1, self.num_envs, dtype=torch.long, device=self.device)
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        self.body_states[env_ids] = self.initial_body_states[env_ids]
        # self.root_states[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        # self.root_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        # self.root_states[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_vec_root_tensor),
                                                     gymtorch.unwrap_tensor(actor_indices), num_resets)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_dof_states),
                                              gymtorch.unwrap_tensor(actor_indices),
                                              num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        return actor_indices

    def pre_physics_step(self, actions):
        self.set_targets()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)

        target_actor_indices = self.all_actor_indices[:, 1].flatten()
        reset_indices = torch.unique(torch.cat([target_actor_indices, actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor,
                                                         gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        # actions[:, 3::] = actions[:, 0:3]
        self.actions = torch.cat((actions, actions), 1).to(self.device).clone()
        # self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):

        if self.time_step == self.max_episode_length:
            self.time_step += 1
            np.save(self.figure_path + 'dof_pos', self.output_dof_pos_tensor.cpu().numpy())
            np.save(self.figure_path + 'dof_vel', self.output_dof_vel_tensor.cpu().numpy())
            np.save(self.figure_path + 'dof_torque', self.output_actuator_force_tensor.cpu().numpy())
            np.save(self.figure_path+'cop',self.cop_tensor.cpu().numpy())

            dof_pos = np.load(self.figure_path + 'dof_pos.npy')  # env_num,dof_num,epoch_length
            dof_vel = np.load(self.figure_path + 'dof_vel.npy')  # env_num,dof_num,epoch_length

            dof_torque = np.load(self.figure_path + 'dof_torque.npy')
            print(dof_pos.shape)
            print(dof_torque.shape)

            peak_phase_difference_list = []
            valley_phase_difference_list = []
            integrated_phase_difference_list = []
            peak_point_hip_list = []
            peak_point_ankle_list = []
            valley_point_hip_list = []
            valley_point_ankle_list = []
            for i in range(dof_pos.shape[0]):
                data = dof_pos[i, :]
                period = int(1205 / (i + 1)) - 1
                peak_point_hip = np.argmax(data[0, int(period / 2) * (i + 1):int(period / 2) * (i + 1) + period]) + int(
                    period / 2) * (i + 1)
                peak_point_hip_list.append(peak_point_hip)
                peak_point_ankle = np.argmax(
                    data[1, int(peak_point_hip - period / 2):int(peak_point_hip + period / 2)]) + int(
                    peak_point_hip - period / 2)
                peak_point_ankle_list.append(peak_point_ankle)
                valley_point_hip = np.argmin(
                    data[0, int(period / 2) * (i + 1):int(period / 2) * (i + 1) + period]) + int(period / 2) * (i + 1)
                valley_point_hip_list.append(valley_point_hip)
                valley_point_ankle = np.argmin(
                    data[1, int(valley_point_hip - period / 2):int(valley_point_hip + period / 2)]) + int(
                    valley_point_hip - period / 2)
                valley_point_ankle_list.append(valley_point_ankle)
                peak_phase_difference = abs(peak_point_hip - peak_point_ankle)
                peak_phase_difference_list.append(peak_phase_difference / period)
                valley_phase_difference = abs(valley_point_hip - valley_point_ankle)
                valley_phase_difference_list.append(valley_phase_difference / period)
                integrated_phase_difference = (peak_phase_difference / period + valley_phase_difference / period) / 2
                integrated_phase_difference_list.append(integrated_phase_difference)
            # dof_pos.shape[0]=9,data.shape[0]=1000

            fig, axs = plt.subplots(dof_pos.shape[0], 1, sharex=True)
            label_list = ['right_hip', "right_ankle", "left_hip", "left_ankle"]
            for i in range(dof_pos.shape[0]):
                data = dof_pos[i, :]
                x0 = peak_point_hip_list[i]
                x1 = peak_point_ankle_list[i]
                x2 = valley_point_hip_list[i]
                x3 = valley_point_ankle_list[i]
                axs[i].plot(x0, data[0][int(x0)], 'ro')
                axs[i].plot(x1, data[1][int(x1)], 'ro')
                axs[i].plot(x2, data[0][int(x2)], 'ro')
                axs[i].plot(x3, data[1][int(x3)], 'ro')
                for j in range(data.shape[0]):
                    axs[i].plot(range(len(data[j])), data[j], label=label_list[j])
                    # axs[i].plot(range(len(data[j])), data[j])

            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            fig.suptitle(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "dof_pos.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            ##corrcoef between ankle and hip
            ankle_hip_corrcoef_list = []
            for i in range(dof_pos.shape[0]):
                data = dof_pos[i, :]
                ankle_hip_corrcoef = np.corrcoef(data[0:2])
                ankle_hip_corrcoef_list.append(ankle_hip_corrcoef[0][-1])
            plt.plot(range(self.dof_pos.shape[0]), ankle_hip_corrcoef_list, marker='*')
            plt.xlabel('Frequency')
            plt.ylabel('ankle_hip_corrcoef')
            for x1, y1 in zip(range(self.dof_pos.shape[0]), ankle_hip_corrcoef_list):
                plt.text(x1, y1 + 0.0001, '%.4f' % y1, ha='center', va='bottom', fontsize=10, rotation=0)
            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            plt.title(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "ankle_hip_corrcoef.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            fig, axs = plt.subplots(self.distance_tensor.shape[0], 1, sharex=True)

            for i in range(self.distance_tensor.shape[0]):
                target_position = self.target_pos_tensor[i, :]
                head_position = self.head_pos_tensor[i, :]
                distance = self.distance_tensor[i, :]
                axs[i].plot(range(self.max_episode_length), target_position, label='target position')
                axs[i].plot(range(self.max_episode_length), head_position, label="head position")
                axs[i].plot(range(self.max_episode_length), distance, label="distance")

            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            fig.suptitle(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "target_head_distance.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            # corrcoef between head and target
            head_target_corrcoef_list = []
            for i in range(self.distance_tensor.shape[0]):
                target_position = self.target_pos_tensor[i, :]
                head_position = self.head_pos_tensor[i, :]
                corr_tensor = torch.stack((target_position, head_position))
                head_target_corrcoef = torch.corrcoef(corr_tensor)
                head_target_corrcoef_list.append(head_target_corrcoef[0][-1].item())

                # target_position = self.target_pos_tensor[-3, :]
                # head_position = self.head_pos_tensor[-3, :]
                target_position_peak_index = (np.diff(np.sign(np.diff(target_position))) < 0).nonzero()[0] + 1
                target_amplitude = []
                head_amplitude = []


                if i == self.distance_tensor.shape[0]-3:

                    for j in range(target_position_peak_index.shape[0]):
                        target_amplitude.append(
                            torch.max(
                                target_position[target_position_peak_index[j]:target_position_peak_index[j + 1]]) -
                            torch.min(target_position[target_position_peak_index[j]:target_position_peak_index[j + 1]]))
                        head_amplitude.append(
                            torch.max(head_position[target_position_peak_index[j]:target_position_peak_index[j + 1]]) -
                            torch.min(head_position[target_position_peak_index[j]:target_position_peak_index[j + 1]]))
                        if j == 11:
                            break

                    print(self.cfg['checkpoint'])
                    print("head_velocity", torch.mean(torch.abs(self.head_vel_tensor[i, :])))
                    print(np.mean(target_amplitude))
                    print("head amplitude", np.mean(head_amplitude))
                    # print(np.var(target_amplitude))
                    # print(np.var(head_amplitude))
                    torque = dof_torque[i, :]
                    vel = dof_vel[i, :]
                    energy = np.sum(np.abs(torque * vel)) * 0.0166
                    print("energy", energy)

                    print(head_target_corrcoef_list[-1])
                    print(head_target_corrcoef_list[-1])
                    print(head_target_corrcoef_list[-1])
                    print(head_target_corrcoef_list[-1])
                    print(head_target_corrcoef_list[-1])
            plt.plot(range(self.distance_tensor.shape[0]), head_target_corrcoef_list, marker='*')
            plt.xlabel('Frequency')
            plt.ylabel('target_head_corrcoef')
            for x1, y1 in zip(range(self.distance_tensor.shape[0]), head_target_corrcoef_list):
                plt.text(x1, y1 + 0.0001, '%.4f' % y1, ha='center', va='bottom', fontsize=10, rotation=0)
            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            plt.title(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "target_head_corrcoef.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            # RMS
            RMS_list = []
            for i in range(dof_pos.shape[0]):
                data = dof_pos[i, :]
                RMS = np.sqrt(np.mean(data[0:2] ** 2))
                RMS_list.append(RMS)
            plt.plot(range(self.dof_pos.shape[0]), RMS_list, marker='*')
            plt.xlabel('Frequency')
            plt.ylabel('RMS')
            for x1, y1 in zip(range(self.dof_pos.shape[0]), RMS_list):
                plt.text(x1, y1 + 0.0001, '%.4f' % y1, ha='center', va='bottom', fontsize=10, rotation=0)
            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            plt.title(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "RMS.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            # relative phase
            relative_phase_list = []
            for i in range(dof_pos.shape[0]):
                data = dof_pos[i, :]
                relative_phase = np.abs(np.abs(data[0]) - np.abs(data[1]))
                relative_phase_list.append(np.max(relative_phase))
            plt.plot(range(self.dof_pos.shape[0]), relative_phase_list, marker='*')
            plt.xlabel('Frequency')
            plt.ylabel('relative_phase')
            for x1, y1 in zip(range(self.dof_pos.shape[0]), relative_phase_list):
                plt.text(x1, y1 + 0.0001, '%.4f' % y1, ha='center', va='bottom', fontsize=10, rotation=0)
            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            plt.title(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "relative_phase.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()

            # energy
            energy_list = []
            for i in range(dof_pos.shape[0]):
                torque = dof_torque[i, :]
                vel = dof_vel[i, :]
                energy = np.sum(np.abs(torque * vel)) * 0.0166
                energy_list.append(np.max(energy))
            plt.plot(range(self.dof_pos.shape[0]), energy_list, marker='*')
            plt.xlabel('Frequency')
            plt.ylabel('energy')
            for x1, y1 in zip(range(self.dof_pos.shape[0]), energy_list):
                plt.text(x1, y1 + 0.0001, '%.4f' % y1, ha='center', va='bottom', fontsize=10, rotation=0)
            title_str = self.cfg['checkpoint'].strip('.pth').split('/')[-1]
            plt.title(title_str)
            plt.legend()
            # plt.savefig(self.figure_path + "energy.png", format="png", dpi=300)
            # plt.show()
            plt.cla()
            plt.close()


            # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
            # x = np.arange(len(labels))  # the label locations
            # width = 0.2  # the width of the bars
            # fig, ax = plt.subplots()
            # rects1 = ax.bar(x - width, valley_phase_difference_list, width, label='Valley')
            # rects2 = ax.bar(x, peak_phase_difference_list, width, label='Peak')
            # rects3 = ax.bar(x + width, integrated_phase_difference_list, width, label='Intergrated')
            # # Add some text for labels, title and custom x-axis tick labels, etc.
            # ax.set_ylabel('Phase Differences')
            # ax.set_xlabel('Frequency')
            # title = str(sum(integrated_phase_difference_list))
            # ax.set_title(title)
            # ax.set_xticks(x, labels)
            # ax.legend()
            # ax.bar_label(rects1, padding=1)
            # ax.bar_label(rects2, padding=1)
            # ax.bar_label(rects3, padding=1)
            # fig.suptitle(title_str)
            # fig.tight_layout()
            # plt.savefig(self.figure_path + "phase_difference.png", format="png", dpi=300)

            stiff_info = self.cfg['experiment'].split('_')
            ankle_hip_corrcoef_arrays = [np.array(['ankle_hip_corrcoef'])]
            energy_arrays = [np.array(['energy'])]
            relative_phase_arrays = [np.array(['relative_phase'])]
            RMS_arrays = [np.array(['RMS'])]
            target_head_corrcoef_arrays = [np.array(['target_head_corrcoef'])]

            for info in stiff_info:
                ankle_hip_corrcoef_arrays.append(np.array([info]))
                energy_arrays.append(np.array([info]))
                relative_phase_arrays.append(np.array([info]))
                RMS_arrays.append(np.array([info]))
                target_head_corrcoef_arrays.append(np.array([info]))

            for i in range(self.num_envs):
                freq_str = 'test_freq' + str(i + 1)
                ankle_hip_corrcoef_arrays_index = ankle_hip_corrcoef_arrays.copy()
                ankle_hip_corrcoef_arrays_index.append(np.array([freq_str]))
                energy_arrays_index = energy_arrays.copy()
                energy_arrays_index.append(np.array([freq_str]))
                relative_phase_arrays_index = relative_phase_arrays.copy()
                relative_phase_arrays_index.append(np.array([freq_str]))
                RMS_arrays_index = RMS_arrays.copy()
                RMS_arrays_index.append(np.array([freq_str]))
                target_head_corrcoef_arrays_index = target_head_corrcoef_arrays.copy()
                target_head_corrcoef_arrays_index.append(np.array([freq_str]))


                ankle_hip_corrcoef_pd = pd.Series(ankle_hip_corrcoef_list[i], index=ankle_hip_corrcoef_arrays_index)
                energy_pd = pd.Series(energy_list[i], index=energy_arrays_index)
                relative_phase_pd = pd.Series(relative_phase_list[i], index=relative_phase_arrays_index)
                RMS_pd = pd.Series(RMS_list[i], index=RMS_arrays_index)
                target_head_corrcoef_pd = pd.Series(head_target_corrcoef_list[i],
                                                    index=target_head_corrcoef_arrays_index)


                all_pd = pd.concat(
                    [ankle_hip_corrcoef_pd, energy_pd, relative_phase_pd, RMS_pd, target_head_corrcoef_pd])

                # all_pd.to_csv(r'/home/li/isaacgym/python/IsaacGymEnvs/isaacgymenvs/runs/all_data.csv', mode='a',
                #               header=False)

            sys.exit()

        elif self.time_step < self.max_episode_length:
            self.output_dof_pos_tensor[..., self.time_step] = self.dof_pos.clone()
            self.output_dof_vel_tensor[..., self.time_step] = self.dof_vel.clone()
            self.output_actuator_force_tensor[
                ..., self.time_step] = self.actions.clone() * self.motor_efforts.unsqueeze(0) * self.power_scale
            # print(self.time_step)

            head_pos = self.body_states[:, 1, 0:3].view(-1, 3)
            head_vel = self.body_states[:, 1, 7:10].view(-1, 3)
            target_dist = torch.sqrt(torch.square(self.marker_positions[:, 0] - head_pos[:, 0]).view(-1))

            self.target_pos_tensor[:, self.time_step] = self.marker_positions[:, 0].clone()
            self.head_pos_tensor[:, self.time_step] = head_pos[:, 0].clone()
            self.head_vel_tensor[:, self.time_step] = head_vel[:, 0].clone()
            self.distance_tensor[:, self.time_step] = target_dist.clone()

            left_front = self.vec_contact_force_tensor[:, 7, 2]
            left_back = self.vec_contact_force_tensor[:, 8, 2]
            right_front = self.vec_contact_force_tensor[:, 12, 2]
            right_back = self.vec_contact_force_tensor[:, 13, 2]
            self.cop = ((left_front + right_front) * 0.12 - (left_back + right_back) * 0.05) / (
                        left_back + left_front + right_front + right_back)
            self.cop_tensor[:, self.time_step] = self.cop.clone()
            # print(self.vec_contact_force_tensor)

            # for i in range(self.num_envs):
            #     actor_handle=self.actor_handles[i]
            #     env=self.envs[i]
            #     rigid_body_properties=self.gym.get_actor_rigid_body_properties(env,actor_handle)
            #     com_x_list=[]
            #     com_y_list=[]
            #     com_z_list=[]
            #     mass_list=[]
            #     for j in rigid_body_properties:
            #         mass_list.append(j.mass)
            #         com_x_list.append(j.com.x)
            #         com_y_list.append(j.com.y)
            #         com_z_list.append(j.com.z)
            #     com_x=sum([a*b for a,b in zip(com_x_list, mass_list)])/sum(mass_list)
            #     # com_y=sum([a*b for a,b in zip[com_y_list, mass_list]])/sum(mass_list)
            #     com_z=sum([a*b for a,b in zip(com_z_list, mass_list)])/sum(mass_list)
            #     # print(com_x)
            # print(self.mass_matrix_tensor)
            self.time_step += 1

            self.progress_buf += 1

            self.compute_observations()
            self.compute_reward(self.actions)
            # dof_angular_pos = pd.DataFrame(self.dof_pos.numpy())
            # if self.progress_buf == 0:
            #     dof_angular_pos.to_csv('/home/li/ray_results/trained_neural_network/temp/angular_pos.csv',
            #                            header=['right_hip', "right_knee", "right_ankle", "left_hip", "left_knee",
            #                                    "left_ankle"],
            #                            index=False)
            # else:
            #     dof_angular_pos.to_csv('/home/li/ray_results/trained_neural_network/temp/angular_pos.csv',
            #                            mode='a', header=False, index=False)

            # debug viz
            if self.viewer and self.debug_viz:
                self.gym.clear_lines(self.viewer)
                points = []
                colors = []
                for i in range(self.num_envs):
                    origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append(
                    [glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                     glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                     glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

                self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_reward(
        marker_pos,  # tensor
        body_states,  # tensor
        initial_body_pos_p,  # tensor
        obs_buf,  # tensor
        reset_buf,  # tensor
        progress_buf,  # tensor
        actions,  # tensor
        up_weight,  # float
        heading_weight,  # float
        actions_cost_scale,  # float
        energy_cost_scale,  # float
        joints_at_limit_cost_scale,  # float
        max_motor_effort,  # float
        motor_efforts,  # tensor
        termination_height,  # float
        death_cost,  # float
        max_episode_length  # float
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, Tensor, float, float, float) -> Tuple[Tensor, Tensor]

    # distance to target
    head_pos = body_states[:, 1, 0:3].view(-1, 3)
    target_dist = torch.sqrt(torch.square(marker_pos[:, 0] - head_pos[:, 0]).view(-1))
    # print(target_dist)
    target_reward = 4.0 / (1.0 + 4 * target_dist.sum() * target_dist.sum())

    # reward for stay still
    foot_initial_pos = initial_body_pos_p[:, [6, 11], :]
    foot_pos = body_states[:, [6, 11], 0:3]
    foot_dist = torch.sqrt(torch.square(foot_pos - foot_initial_pos).sum(-1).sum(-1))
    stay_reward = 1.0 / (1.0 + foot_dist * foot_dist)

    # reward for being upright
    up_reward = torch.ones_like(obs_buf[:, 10])
    up_reward = torch.where(obs_buf[:, 10] > 1.2, up_reward + up_weight, up_reward)

    actions_cost = torch.sum(actions ** 2, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 11:15]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 11:15]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0),
                                  dim=-1)

    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 15:19]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    # stay_reward + target_reward
    total_reward = stay_reward + target_reward + up_reward - actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost,
                               total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(foot_dist >= 0.1, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # reset = torch.where(target_dist >= 0.04, torch.ones_like(reset_buf), reset)
    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(
        body_states,  # tensor
        root_states,  # tensor
        inv_start_rot,  # tensor
        dof_pos,  # tensor
        dof_vel,  # tensor
        dof_force,  # tensor
        dof_limits_lower,  # tensor
        dof_limits_upper,  # tensor
        dof_vel_scale,  # float
        sensor_force_torques,  # tensor
        actions,  # tensor
        contact_force_scale,  # float
        angular_velocity_scale,  # float
        basis_vec1,  # tensor
        marker_pos  # tensor
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    head_pos_vel_x = body_states[:, 1, [0, 7]].view(-1, 2)

    foot_pos = body_states[:, [6, 11], 0:3].view(-1, 6)

    torso_quat, up_proj, up_vec = compute_up_swing(
        torso_rotation, inv_start_rot, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_swing(
        torso_quat, velocity, ang_velocity)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    obs = torch.cat(
        (torso_position[:, 2].view(-1, 1),
         vel_loc,
         angvel_loc * angular_velocity_scale,
         yaw,
         roll,
         up_proj.unsqueeze(-1),  # 10
         dof_pos_scaled,
         dof_vel * dof_vel_scale,
         dof_force * contact_force_scale,
         sensor_force_torques.view(-1, 12) * contact_force_scale,
         head_pos_vel_x,
         foot_pos,
         actions,
         marker_pos[:, 0].view(-1, 1)),
        dim=-1)

    return obs, up_vec
