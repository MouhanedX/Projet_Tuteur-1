"""Capture raw CARLA frames with an ego car, pedestrians and 43 sign classes."""
import json
import math
import os
import queue
import random

import carla
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_frames")
MANIFEST = os.path.join(BASE_DIR, "capture_manifest.jsonl")
W, H, FOV = 960, 540, 90.0
FX = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
WEATHERS = ("ClearNoon", "CloudySunset", "SoftRainNoon")
CLASS_SCENARIOS = (tuple(range(0, 14)), tuple(range(14, 28)), tuple(range(28, 43)))
MIN_DISTANCE, MAX_DISTANCE, MIN_PIXELS = 8.0, 45.0, 14.0
FACE_Z, HALF_WIDTH = 1.30, 0.46
CAPTURE_EVERY_TICKS = 3


def project(points, camera_transform):
    camera_location = camera_transform.location
    forward_vector = camera_transform.get_forward_vector()
    right_vector = camera_transform.get_right_vector()
    up_vector = camera_transform.get_up_vector()
    delta = points - np.array((
        camera_location.x,
        camera_location.y,
        camera_location.z,
    ))
    forward = delta @ np.array((
        forward_vector.x,
        forward_vector.y,
        forward_vector.z,
    ))
    if (forward <= 0.05).any():
        return None
    right = delta @ np.array((right_vector.x, right_vector.y, right_vector.z))
    up = delta @ np.array((up_vector.x, up_vector.y, up_vector.z))
    return np.stack((W / 2 + FX * right / forward, H / 2 - FX * up / forward), 1)


def make_route(world):
    spawn = world.get_map().get_spawn_points()[0]
    route = [world.get_map().get_waypoint(spawn.location)]
    for _ in range(320):
        next_waypoints = route[-1].next(5.0)
        if not next_waypoints:
            break
        route.append(next_waypoints[0])
    return spawn, route


def spawn_signs(world, route, classes):
    blueprint = world.get_blueprint_library().find("static.prop.streetsign")
    candidates = []
    last_index = -100
    for index, waypoint in zip(range(0, len(route), 5), route[::5]):
        if index - last_index < 4 or waypoint.is_junction:
            continue
        if any(candidate.is_junction for candidate in waypoint.next(15.0)):
            continue
        yaw = math.radians(waypoint.transform.rotation.yaw)
        right = np.array((math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2), 0.0))
        offset = waypoint.lane_width / 2.0 + 1.2
        location = waypoint.transform.location + carla.Location(
            x=float(right[0] * offset), y=float(right[1] * offset)
        )
        ground = world.cast_ray(
            carla.Location(location.x, location.y, location.z + 2),
            carla.Location(location.x, location.y, location.z - 2),
        )
        if not ground:
            continue
        transform = carla.Transform(
            carla.Location(location.x, location.y, ground[0].location.z + 0.5),
            carla.Rotation(yaw=(waypoint.transform.rotation.yaw + 90) % 360),
        )
        candidates.append(transform)
        last_index = index
    signs = []
    if len(candidates) < len(classes):
        raise RuntimeError(
            f"Could validate only {len(candidates)} sign locations; "
            f"{len(classes)} are required for this scenario."
        )
    for cls, transform in zip(classes, candidates):
        actor = world.try_spawn_actor(blueprint, transform)
        if actor:
            yaw = math.radians(transform.rotation.yaw)
            horizontal = np.array((math.cos(yaw), math.sin(yaw), 0.0)) * HALF_WIDTH
            signs.append({
                "actor": actor,
                "class": cls,
                "center": np.array((transform.location.x, transform.location.y,
                                    transform.location.z + FACE_Z)),
                "horizontal": horizontal,
            })
    if len(signs) != len(classes):
        raise RuntimeError(
            f"Could place only {len(signs)} validated signs; {len(classes)} are required. "
            "Use a longer route or a CARLA map with more roadside clearance."
        )
    return signs


def spawn_pedestrians(world, count=12):
    walkers, controllers = [], []
    walker_bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    for _ in range(count):
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(location))
        if walker:
            walkers.append(walker)
            controller = world.try_spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
            if controller:
                controllers.append(controller)
    for controller in controllers:
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination:
            controller.go_to_location(destination)
        controller.set_max_speed(1.4)
    world.tick()
    return walkers, controllers


def destroy_actors(world):
    for pattern in ("static.prop.streetsign*", "vehicle.*", "sensor.*", "walker.*"):
        for actor in world.get_actors().filter(pattern):
            try:
                actor.destroy()
            except RuntimeError:
                pass


def follow_vehicle(world, vehicle):
    vehicle_transform = vehicle.get_transform()
    forward = vehicle_transform.get_forward_vector()
    spectator_location = vehicle_transform.location + carla.Location(
        x=-forward.x * 12.0,
        y=-forward.y * 12.0,
        z=8.0,
    )
    spectator_rotation = carla.Rotation(
        pitch=-25.0,
        yaw=vehicle_transform.rotation.yaw,
        roll=0.0,
    )
    world.get_spectator().set_transform(
        carla.Transform(spectator_location, spectator_rotation)
    )


def main():
    random.seed(7)
    os.makedirs(RAW_DIR, exist_ok=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    actors = []
    try:
        destroy_actors(world)
        world.tick()
        spawn, route = make_route(world)
        vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        follow_vehicle(world, vehicle)
        walkers, controllers = spawn_pedestrians(world)
        actors.extend(walkers + controllers)
        print(f"Spawned vehicle and {len(walkers)} pedestrians", flush=True)
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(W))
        camera_bp.set_attribute("image_size_y", str(H))
        camera_bp.set_attribute("fov", str(FOV))
        camera_local = carla.Transform(carla.Location(x=0.5, z=1.9))
        camera = world.spawn_actor(camera_bp, camera_local, attach_to=vehicle)
        actors.append(camera)
        frames = queue.Queue()
        camera.listen(frames.put)
        with open(MANIFEST, "w", encoding="utf-8") as manifest:
            frame_id = 0
            for scenario_id, classes in enumerate(CLASS_SCENARIOS, start=1):
                signs = spawn_signs(world, route, classes)
                print(
                    f"Scenario {scenario_id}/{len(CLASS_SCENARIOS)}: "
                    f"{len(signs)} signs, classes {classes[0]}-{classes[-1]}",
                    flush=True,
                )
                for sign in signs:
                    actors.append(sign["actor"])
                print(
                    f"Actor IDs: {[sign['actor'].id for sign in signs]}",
                    flush=True,
                )
                for weather_name in WEATHERS:
                    world.set_weather(getattr(carla.WeatherParameters, weather_name))
                    vehicle.set_transform(spawn)
                    vehicle.set_target_velocity(carla.Vector3D())
                    follower_index = 0
                    tick_count = 0
                    rejected = {"behind": 0, "distance": 0, "pixels": 0, "frame": 0}
                    while follower_index < len(route) - 3 and frame_id < 12000:
                        transform = vehicle.get_transform()
                        velocity = vehicle.get_velocity()
                        speed = math.hypot(velocity.x, velocity.y)
                        target = route[min(follower_index + 1, len(route) - 1)].transform.location
                        angle = math.degrees(math.atan2(
                            target.y - transform.location.y,
                            target.x - transform.location.x,
                        ))
                        error = (angle - transform.rotation.yaw + 540) % 360 - 180
                        control = carla.VehicleControl(
                            steer=max(-0.6, min(0.6, error / 60))
                        )
                        control.throttle = 0.55 if speed < 5.6 else 0.0
                        control.brake = 0.2 if speed >= 5.6 else 0.0
                        vehicle.apply_control(control)
                        world.tick()
                        follow_vehicle(world, vehicle)
                        tick_count += 1
                        if follower_index < len(route) - 1:
                            here = transform.location
                            next_location = route[follower_index + 1].transform.location
                            if here.distance(next_location) < 7:
                                follower_index += 1
                        if tick_count % CAPTURE_EVERY_TICKS:
                            continue
                        image = frames.get(timeout=10)
                        frame = np.frombuffer(image.raw_data, np.uint8).reshape(
                            H, W, 4
                        )[:, :, :3].copy()
                        # Use the sensor's world transform directly. Rebuilding
                        # it from vehicle and local matrices can invert the
                        # camera basis depending on the CARLA client version.
                        camera_transform = camera.get_transform()
                        visible = []
                        for sign in signs:
                            quad3d = sign["center"] + np.array([
                                -sign["horizontal"] + (0, 0, HALF_WIDTH),
                                sign["horizontal"] + (0, 0, HALF_WIDTH),
                                sign["horizontal"] - (0, 0, HALF_WIDTH),
                                -sign["horizontal"] - (0, 0, HALF_WIDTH),
                            ])
                            quad = project(quad3d, camera_transform)
                            distance = float(np.linalg.norm(
                                sign["center"] - np.array((
                                    camera_transform.location.x,
                                    camera_transform.location.y,
                                    camera_transform.location.z,
                                ))
                            ))
                            if quad is None:
                                rejected["behind"] += 1
                                continue
                            if not MIN_DISTANCE <= distance <= MAX_DISTANCE:
                                rejected["distance"] += 1
                                continue
                            if np.ptp(quad[:, 0]) < MIN_PIXELS:
                                rejected["pixels"] += 1
                                continue
                            if (quad[:, 0].min() < 2 or quad[:, 1].min() < 2 or
                                    quad[:, 0].max() >= W - 2 or
                                    quad[:, 1].max() >= H - 2):
                                rejected["frame"] += 1
                                continue
                            visible.append({
                                "class": sign["class"],
                                "distance_m": round(distance, 2),
                                "quad": quad.round(3).tolist(),
                            })
                        if visible:
                            filename = f"scenario{scenario_id}_{weather_name}_{frame_id:06d}.png"
                            cv2.imwrite(os.path.join(RAW_DIR, filename), frame)
                            manifest.write(json.dumps({
                                "file": filename,
                                "scenario": scenario_id,
                                "weather": weather_name,
                                "signs": visible,
                            }) + "\n")
                            manifest.flush()
                            frame_id += 1
                        if follower_index >= len(route) - 3:
                            break
                    print(
                        f"{weather_name}: ticks={tick_count}, route_index={follower_index}, "
                        f"frames={frame_id}, rejected={rejected}",
                        flush=True,
                    )
                for sign in signs:
                    sign["actor"].destroy()
                    actors.remove(sign["actor"])
        print(
            f"Captured {frame_id} raw frames across {len(CLASS_SCENARIOS)} "
            f"scenarios and {WEATHERS} weather settings to {RAW_DIR}"
        )
    finally:
        for actor in reversed(actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original)


if __name__ == "__main__":
    main()
