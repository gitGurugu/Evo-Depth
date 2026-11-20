"""Per-control-step metrics shared by the LIBERO and LIBERO-PLUS clients."""

import json
import math
from pathlib import Path


PICK_LIFT_THRESHOLD_M = 0.03
MAX_CONTACT_PAIRS = 20
SIM_API_ERRORS = (AttributeError, KeyError, IndexError, TypeError, ValueError)


def object_registry(env):
    objects = getattr(env, "objects_dict", None)
    if objects is None:
        objects = getattr(env, "objects", getattr(env, "_objects", []))
    if not isinstance(objects, dict):
        objects = {obj.name: obj for obj in objects}
    return objects, getattr(env, "fixtures_dict", {})


def resolve_target(env):
    # ControlEnv wraps the robosuite domain in .env (LIBERO-plus env_wrapper.py).
    objects, fixtures = object_registry(env)
    registry = dict(fixtures, **objects)
    problem = getattr(env, "parsed_problem", {})
    subjects = []

    def visit_goals(goals):
        for goal in goals:
            if not isinstance(goal, (list, tuple)) or not goal:
                continue
            if goal[0] in ("and", "or", "not"):
                visit_goals(goal[1:])
            elif len(goal) > 1:
                subject = goal[1]
                subject = problem.get("regions", {}).get(subject, {}).get("target", subject)
                if subject in registry and subject not in subjects:
                    subjects.append(subject)

    visit_goals(problem.get("goal_state", []))
    candidates = [name for name in subjects if name in objects] or subjects
    source = "first_goal_subject"
    if not candidates:
        candidates = [name for name in getattr(env, "obj_of_interest", []) if name in registry]
        candidates = [name for name in candidates if name in objects] or candidates
        source = "first_object_of_interest"
    if not candidates and len(objects) == 1:
        candidates = list(objects)
        source = "only_movable_object"
    return {
        "name": candidates[0] if candidates else None,
        "candidates": candidates,
        "selection": source if candidates else "unresolved",
        "ambiguous": len(candidates) > 1,
    }


def object_position(env, name):
    if name is None:
        return None
    try:
        body_id = getattr(env, "obj_body_id", {}).get(name)
        if body_id is None:
            objects, fixtures = object_registry(env)
            obj = objects.get(name, fixtures.get(name))
            body_id = env.sim.model.body_name2id(obj.root_body)
        if body_id < 0:
            return None
        return [float(value) for value in env.sim.data.body_xpos[body_id]]
    except SIM_API_ERRORS:
        return None


def read_contacts(env, target_name):
    result = {
        "contacts_available": False,
        "collision_count": None,
        "collision_pairs": [],
        "gripper_target_contact": None,
        "target_supported": None,
    }
    try:
        robot_geoms = set()
        gripper_geoms = set()
        for robot in env.robots:
            robot_geoms.update(robot.robot_model.contact_geoms)
            grippers = robot.gripper
            if not isinstance(grippers, dict):
                grippers = {"arm": grippers}
            for gripper in grippers.values():
                gripper_geoms.update(gripper.contact_geoms)
        robot_geoms.update(gripper_geoms)
        if not robot_geoms:
            return result

        objects, fixtures = object_registry(env)
        target = objects.get(target_name, fixtures.get(target_name))
        target_geoms = set(getattr(target, "contact_geoms", []))
        count = 0
        pairs = []
        gripper_contact = False
        supported = False
        sim = env.sim
        for index in range(sim.data.ncon):
            contact = sim.data.contact[index]
            first = sim.model.geom_id2name(contact.geom1) or f"geom#{contact.geom1}"
            second = sim.model.geom_id2name(contact.geom2) or f"geom#{contact.geom2}"
            if (first in robot_geoms) != (second in robot_geoms):
                count += 1
                pair = sorted([first, second])
                if len(pairs) < MAX_CONTACT_PAIRS and pair not in pairs:
                    pairs.append(pair)
            if (first in target_geoms) != (second in target_geoms):
                other = second if first in target_geoms else first
                gripper_contact |= other in gripper_geoms
                supported |= other not in robot_geoms
        result.update(
            contacts_available=True,
            collision_count=count,
            collision_pairs=pairs,
            gripper_target_contact=gripper_contact if target_geoms and gripper_geoms else None,
            target_supported=supported if target_geoms else None,
        )
    except SIM_API_ERRORS as error:
        result["contact_error"] = type(error).__name__
    return result


def new_episode_metrics(env, **episode):
    domain = getattr(env, "env", env)
    target = resolve_target(domain)
    return {
        "schema_version": 1,
        **episode,
        "coordinate_frame": "MuJoCo world; meters; object root body origin",
        "target_object": target,
        "initial_object_xyz": object_position(domain, target["name"]),
        "pick_lift_threshold_m": PICK_LIFT_THRESHOLD_M,
        "max_contact_pairs": MAX_CONTACT_PAIRS,
        "steps": [],
    }


def record_step(metrics, env, obs, action, phase="policy"):
    domain = getattr(env, "env", env)
    target_xyz = object_position(domain, metrics["target_object"]["name"])
    ee_xyz = obs.get("robot0_eef_pos")
    ee_xyz = [float(value) for value in ee_xyz] if ee_xyz is not None else None
    distance = None
    if ee_xyz is not None and target_xyz is not None:
        distance = math.sqrt(sum((ee - obj) ** 2 for ee, obj in zip(ee_xyz, target_xyz)))
    contacts = read_contacts(domain, metrics["target_object"]["name"])
    initial_xyz = metrics["initial_object_xyz"]
    lift = target_xyz[2] - initial_xyz[2] if target_xyz is not None and initial_xyz is not None else None
    close_command = float(action[6]) > 0  # robosuite: -1 opens, +1 closes.
    picked = None
    if lift is not None and contacts["gripper_target_contact"] is not None:
        # shortcut: contact + 3 cm lift is a grasp heuristic; use task-specific checks for grasp benchmarks.
        picked = bool(phase == "policy" and close_command and lift >= PICK_LIFT_THRESHOLD_M
                      and contacts["gripper_target_contact"] and contacts["target_supported"] is False)
    metrics["steps"].append({
        "step": len(metrics["steps"]),
        "phase": phase,
        "action": [float(value) for value in action],
        "ee_xyz": ee_xyz,
        "target_object_xyz": target_xyz,
        "target_unresolved": target_xyz is None,
        "gripper_object_distance": distance,
        "object_lift_m": lift,
        "gripper_close_command": close_command,
        "pick_success": picked,
        **contacts,
    })


def save_episode_metrics(metrics, video_path, episode_success):
    policy_steps = [step for step in metrics["steps"] if step["phase"] == "policy"]
    picks = [step["pick_success"] for step in policy_steps]
    pick_success = None
    if any(picks):
        pick_success = True
    elif picks and None not in picks:
        pick_success = False
    counts = [step["collision_count"] for step in policy_steps]
    distances = [step["gripper_object_distance"] for step in policy_steps
                 if step["gripper_object_distance"] is not None]
    summary = {
        "episode_success": bool(episode_success),
        "pick_success": pick_success,
        "collision_count": sum(counts) if counts and all(count is not None for count in counts) else None,
        "min_distance": min(distances) if distances else None,
    }
    metrics["summary"] = summary
    # Sidecars share the existing video basename, including suite/category directories.
    path = Path(video_path).with_suffix(".metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return summary
