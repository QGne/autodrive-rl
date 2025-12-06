# envs/carla_lane_env.py

import math
import time
from typing import Tuple, Dict, Any

import numpy as np
import carla


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class CarlaLaneEnv:
    """
    Minimal Gym-style environment for lane-following / route-progress in CARLA.

    State: [lane_offset (m), heading_error (rad), speed (m/s), normalized_step]
    Actions (discrete):
        0 -> steer left
        1 -> steer straight
        2 -> steer right
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town_name: str = "Town03",  # pick any town you have
        fixed_delta_seconds: float = 0.05,
        max_steps_per_episode: int = 500,
    ):
        self.host = host
        self.port = port
        self.town_name = town_name
        self.fixed_delta_seconds = fixed_delta_seconds
        self.max_steps = max_steps_per_episode

        # action space: 3 discrete steering commands
        self.action_space_n = 3
        # state dimension: 4 (lane_offset, heading_error, speed, step_norm)
        self.state_dim = 4

        self.client = None
        self.world = None
        
        self.spectator = None
        self.realtime_render = True      # slow down for human eyes
        self.render_dt = 1.0 / 20.0      # ~20 FPS
        
        self.map = None

        self.vehicle = None
        self.blueprint_library = None
        self.spawn_point = None

        self.current_step = 0

        # steer magnitudes (radians mapped to [-1, 1] range by CARLA)
        self.steer_values = {
            0: -0.3,  # left
            1: 0.0,   # straight
            2: 0.3,   # right
        }

        self._connect_to_carla()
        self._setup_world()

    # ------------- Setup helpers -------------

    def _connect_to_carla(self):
        print(f"[CarlaLaneEnv] Connecting to CARLA at {self.host}:{self.port} ...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)

        # load world if needed
        world = self.client.get_world()
        if self.town_name not in world.get_map().name:
            print(f"[CarlaLaneEnv] Loading town: {self.town_name}")
            world = self.client.load_world(self.town_name)

        self.world = world
        
        self.spectator = self.world.get_spectator()    #attempt to add camera to car
        
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    def _setup_world(self):
        """Enable synchronous mode with fixed delta time."""
        settings = self.world.get_settings()
        if not settings.synchronous_mode:
            print("[CarlaLaneEnv] Enabling synchronous mode.")
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

    # ------------- Core API -------------

    def reset(self) -> np.ndarray:
        """
        Reset environment: spawn vehicle at a random spawn point, zero step counter.
        Returns initial state.
        """
        # destroy previous vehicle if it exists
        self._cleanup_actors()

        # pick a random spawn point
        spawn_points = self.map.get_spawn_points()
        self.spawn_point = np.random.choice(spawn_points)

        vehicle_bp = self.blueprint_library.filter("vehicle.*model3*")
        if vehicle_bp:
            vehicle_bp = vehicle_bp[0]
        else:
            vehicle_bp = self.blueprint_library.filter("vehicle.*")[0]

        self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # let the world tick once to settle
        self.world.tick()
        self.current_step = 0
        
        self._update_spectator()

        # optionally slow down for human viewing on reset
        if self.realtime_render:
            time.sleep(self.render_dt)

        state = self._get_state()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply action (steering), step the world, compute new state and reward.
        Returns: next_state, reward, done, info
        """
        if action not in self.steer_values:
            raise ValueError(f"Invalid action {action}; must be 0,1,2.")

        steer = self.steer_values[action]
        throttle = 0.4  # constant throttle for now
        brake = 0.0

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )
        self.vehicle.apply_control(control)

        # advance the simulation by one tick
        self.world.tick()
        self.current_step += 1
        
        #addition for spectator here
        self._update_spectator()

        # slow down so each step is visible (approx. render_dt real seconds)
        if self.realtime_render:
            time.sleep(self.render_dt)

        next_state = self._get_state()
        reward, done, info = self._compute_reward_done(next_state)

        return next_state, reward, done, info

    # ------------- State & reward -------------

    def _get_state(self) -> np.ndarray:
        """
        Build low-dimensional state:
        [lane_offset (meters), heading_error (radians), speed (m/s), step_norm [0,1]]
        """
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        loc = transform.location

        # get waypoint on nearest driving lane
        waypoint = self.map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # lane center transform
        lane_tf = waypoint.transform

        # compute heading error
        veh_yaw = _deg2rad(transform.rotation.yaw)
        lane_yaw = _deg2rad(lane_tf.rotation.yaw)
        heading_error = _normalize_angle(veh_yaw - lane_yaw)

        # compute lateral offset (project difference onto lane's right vector)
        lane_loc = lane_tf.location
        dx = loc.x - lane_loc.x
        dy = loc.y - lane_loc.y

        lane_forward = lane_tf.get_forward_vector()
        # right vector (perpendicular)
        lane_right = carla.Vector3D(lane_forward.y, -lane_forward.x, 0.0)

        # normalize right vector
        right_norm = math.sqrt(lane_right.x ** 2 + lane_right.y ** 2) + 1e-8
        lane_right.x /= right_norm
        lane_right.y /= right_norm

        lateral_offset = dx * lane_right.x + dy * lane_right.y  # in meters

        # speed magnitude
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        # normalized step index
        step_norm = float(self.current_step) / float(self.max_steps)

        state = np.array(
            [lateral_offset, heading_error, speed, step_norm],
            dtype=np.float32,
        )
        return state
        
    def _update_spectator(self):
        """
        Attach the spectator behind and slightly above the ego vehicle,
        looking forward along the car's heading.
        """
        if self.vehicle is None or self.spectator is None:
            return

        veh_tf = self.vehicle.get_transform()
        loc = veh_tf.location
        rot = veh_tf.rotation

        # Move camera back along the car's forward direction.
        # forward = direction car is facing
        forward = veh_tf.get_forward_vector()
        distance_back = 6.0   # meters behind the car
        height = 2.5          # meters above ground

        cam_loc = carla.Location(
            x=loc.x - forward.x * distance_back,
            y=loc.y - forward.y * distance_back,
            z=loc.z + height,
        )

        cam_rot = carla.Rotation(
            pitch=-10.0,             # tilt slightly down
            yaw=rot.yaw,             # same yaw as car
            roll=0.0,
        )

        self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def _compute_reward_done(self, state: np.ndarray) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Reward function:
            +1 per step
            - alpha * |lateral_offset|
            - beta * |heading_error|
        Episode terminates if:
            - |lateral_offset| > max_offset
            - current_step >= max_steps
            (later we can add collisions, etc.)
        """
        lateral_offset, heading_error, speed, step_norm = state

        alpha = 0.5
        beta = 0.2
        max_offset = 2.0  # meters

        reward = 1.0 - alpha * abs(lateral_offset) - beta * abs(heading_error)
        done = False
        info: Dict[str, Any] = {}

        # terminate if car is too far from lane center
        if abs(lateral_offset) > max_offset:
            done = True
            reward -= 10.0
            info["reason"] = "lane_departure"

        # terminate if time horizon exceeded
        if self.current_step >= self.max_steps:
            done = True
            info["reason"] = info.get("reason", "max_steps")

        return float(reward), done, info

    # ------------- Cleanup -------------

    def _cleanup_actors(self):
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass
            self.vehicle = None

    def close(self):
        """Restore async mode and destroy actors."""
        self._cleanup_actors()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)


if __name__ == "__main__":
    # Simple manual test: random policy for a few steps
    env = CarlaLaneEnv()
    try:
        state = env.reset()
        print("Initial state:", state)
        for t in range(50):
            a = np.random.randint(0, env.action_space_n)
            s2, r, done, info = env.step(a)
            print(f"t={t}, a={a}, r={r:.3f}, done={done}, info={info}")
            if done:
                break
    finally:
        env.close()
        print("Environment closed.")
