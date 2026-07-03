# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import numpy as np
from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Transform3d, Translation3d

from backend.config.config import LocalConfig
from backend.vision_types import CameraPoseObservation, PoseEstimate, RobotPoseEstimation

class RobotPoseEstimator: 
    def __init__(self, local_config: LocalConfig, xy_coeff=0.01, theta_coeff=0.03, ambiguity_threshold=0.4, 
                 field_border_margin=0.5, z_min=-0.5, z_max=1):
        self.local_config = local_config
        self.xy_coeff = xy_coeff
        self.theta_coeff = theta_coeff
        self.ambiguity_threshold = ambiguity_threshold
        self.field_border_margin = field_border_margin
        self.z_min = z_min
        self.z_max = z_max

        self.final_robot_pose: RobotPoseEstimation | None = None
    
    @staticmethod
    def _to_transform3d(pose: Pose3d) -> Transform3d:
        return Transform3d(pose.translation(), pose.rotation())

    def _compute_std_dev(self, avg_distance: float, tag_count: int, use_vision_rot: bool) -> tuple[float, float]:
        base = (avg_distance ** 2) / (tag_count ** 2)

        xyStdDev = self.xy_coeff * base
        thetaStdDev = self.theta_coeff * base if use_vision_rot else np.inf

        return xyStdDev, thetaStdDev
    
    def _process_observation(self, obs: CameraPoseObservation, robot_to_cam: Pose3d) -> PoseEstimate | None:
        inverse: Transform3d = self._to_transform3d(robot_to_cam).inverse()
        use_vision_rot: bool = False
        
        # Multi Tag
        if obs.pose_1 is None:
            robot_pose = obs.pose_0.transformBy(inverse)
            use_vision_rot = True
        else:
            # Single Tag, disambiguate
            robot_pose_0 = obs.pose_0.transformBy(inverse)
            robot_pose_1 = obs.pose_1.transformBy(inverse)

            robot_pose = robot_pose_0 if obs.error_0 < obs.error_1 else robot_pose_1

        field = self.local_config.tag_layout["field"]

        if (robot_pose.X() < -self.field_border_margin 
            or robot_pose.X() > field["length"] + self.field_border_margin
            or robot_pose.Y() < -self.field_border_margin
            or robot_pose.Y() > field["width"] + self.field_border_margin
            or robot_pose.Z() < self.z_min 
            or robot_pose.Z() > self.z_max):
            return None
        
        tag_poses: list[Pose3d] = []
        for tag in obs.tag_ids:
            for tag_data in self.local_config.tag_layout["tags"]:
                if tag_data["ID"] == tag:
                    tag_poses.append(Pose3d(
                        Translation3d(
                            tag_data["pose"]["translation"]["x"],
                            tag_data["pose"]["translation"]["y"],
                            tag_data["pose"]["translation"]["z"],
                        ),
                        Rotation3d(Quaternion(
                            tag_data["pose"]["rotation"]["quaternion"]["W"],
                            tag_data["pose"]["rotation"]["quaternion"]["X"],
                            tag_data["pose"]["rotation"]["quaternion"]["Y"],
                            tag_data["pose"]["rotation"]["quaternion"]["Z"],
                        )),
                    ))
                    break
        
        if len(tag_poses) <= 0:
            return None
        
        avg_dis = sum(
            tp.translation().distance(robot_pose.translation()) for tp in tag_poses
        ) / len(tag_poses)

        xy, theta = self._compute_std_dev(avg_dis, len(tag_poses), use_vision_rot)

        return PoseEstimate(robot_pose, xy, theta)
    
    @staticmethod
    def _fuse(estimates: list[PoseEstimate]) -> PoseEstimate:
        assert len(estimates) > 0

        xy_weights = [1.0 / (e.xy_std_dev ** 2) for e in estimates]
        theta_weights = [
            1.0 / (e.theta_std_dev ** 2) if np.isfinite(e.theta_std_dev) else 0.0
            for e in estimates
        ]
        
        total_xy_weight = sum(xy_weights)
        total_theta_weight = sum(theta_weights)

        fused_x = sum(w * e.pose.translation().X() for w, e in zip(xy_weights, estimates)) / total_xy_weight
        fused_y = sum(w * e.pose.translation().Y() for w, e in zip(xy_weights, estimates)) / total_xy_weight
        fused_z = sum(w * e.pose.translation().Z() for w, e in zip(xy_weights, estimates)) / total_xy_weight

        quats = [e.pose.rotation().getQuaternion() for e in estimates]
        ref = np.array([quats[0].W(), quats[0].X(), quats[0].Y(), quats[0].Z()])
        q_sum = np.zeros(4)
        for w, q in zip(theta_weights, quats):
            arr = np.array([q.W(), q.X(), q.Y(), q.Z()])
            if np.dot(arr, ref) < 0:
                arr = -arr
            q_sum += w * arr
        
        # Fall back to first pose rotation if no estimate has vision rotation
        if np.linalg.norm(q_sum) < 1e-9:
            q_norm = ref
            fused_theta_std_dev = np.inf
        else:
            q_norm = q_sum / np.linalg.norm(q_sum)
            fused_theta_std_dev = 1.0 / np.sqrt(total_theta_weight)

        fused_pose = Pose3d(
            Translation3d(fused_x, fused_y, fused_z), 
            Rotation3d(Quaternion(q_norm[0], q_norm[1], q_norm[2], q_norm[3]))
        )

        fused_xy_std_dev = 1.0 / np.sqrt(total_xy_weight)

        return PoseEstimate(fused_pose, fused_xy_std_dev, fused_theta_std_dev)
    
    def update(self, observations: list[CameraPoseObservation], camera_transforms: list[Pose3d], timestamp: int) -> None:
        pose_ests: list[PoseEstimate] = []

        for i, obs in enumerate(observations):
            if obs is None:
                continue

            result = self._process_observation(obs, camera_transforms[i])

            if result is None:
                continue

            pose_ests.append(result)

        if not pose_ests:
            return
        
        fused = self._fuse(pose_ests)
        self.final_robot_pose = RobotPoseEstimation(
            fused.pose.toPose2d(),
            timestamp,
            fused.xy_std_dev,
            fused.theta_std_dev
        )
        
    def get_last_pose(self):
        return self.final_robot_pose