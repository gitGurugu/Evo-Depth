"""Run with Python's standard library only: python metrics_selfcheck.py."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Fake

from eval_metrics import new_episode_metrics, record_step, save_episode_metrics


def selfcheck():
    names = ["robot0_arm", "gripper0_left", "gripper0_right", "bowl_geom", "table_geom", "plate_geom"]
    bowl = Fake(name="bowl", root_body="bowl_body", contact_geoms=["bowl_geom"])
    plate = Fake(name="plate", root_body="plate_body", contact_geoms=["plate_geom"])
    data = Fake(body_xpos=[[0, 0, 0.8], [0.2, 0, 0.8]], ncon=0, contact=[])
    model = Fake(geom_id2name=names.__getitem__, body_name2id={"bowl_body": 0, "plate_body": 1}.__getitem__)
    domain = Fake(
        sim=Fake(model=model, data=data), objects_dict={"bowl": bowl, "plate": plate},
        fixtures_dict={}, obj_body_id={"bowl": 0, "plate": 1},
        parsed_problem={"goal_state": [["on", "bowl", "plate"]]},
        obj_of_interest=["plate", "bowl"],
        robots=[Fake(robot_model=Fake(contact_geoms=["robot0_arm"]),
                     gripper=Fake(contact_geoms=["gripper0_left", "gripper0_right"]))],
    )
    env = Fake(env=domain)
    obs = {"robot0_eef_pos": [0, 0, 0.9]}
    closed_action = [0.0] * 6 + [1.0]

    def contacts(*pairs):
        data.contact = [Fake(geom1=first, geom2=second) for first, second in pairs]
        data.ncon = len(pairs)

    metrics = new_episode_metrics(env, task_id=0, episode=1)
    assert metrics["target_object"]["name"] == "bowl"  # Goal subject wins over destination.
    contacts((1, 2), (3, 4), (1, 3), (0, 4))  # Self, scene, grasp, arm-scene.
    record_step(metrics, env, obs, [0.0] * 7, phase="settle")
    assert metrics["steps"][0]["collision_count"] == 2
    assert metrics["steps"][0]["pick_success"] is False

    data.body_xpos[0][2] = 0.85
    contacts((1, 3), (2, 3))
    record_step(metrics, env, obs, closed_action)
    lifted = metrics["steps"][-1]
    assert lifted["pick_success"] is True and lifted["target_supported"] is False
    assert abs(lifted["gripper_object_distance"] - 0.05) < 1e-8
    assert lifted["action"] == closed_action and len(lifted["action"]) == 7
    assert metrics["steps"][0]["target_object_xyz"][2] == 0.8  # Snapshot, not a sim view.

    contacts((1, 3), (3, 4))
    record_step(metrics, env, obs, closed_action)
    assert metrics["steps"][-1]["pick_success"] is False  # Elevated but supported.
    contacts((1, 3))
    record_step(metrics, env, obs, [0.0] * 6 + [-1.0])
    assert metrics["steps"][-1]["pick_success"] is False  # Open command.
    contacts()
    record_step(metrics, env, obs, closed_action)
    assert metrics["steps"][-1]["pick_success"] is False  # Airborne without grasp contact.

    with TemporaryDirectory() as directory:
        video = Path(directory) / "suite" / "category" / "task0_episode1.mp4"
        summary = save_episode_metrics(metrics, video, False)
        saved = json.loads(video.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        assert saved == metrics and summary["pick_success"] is True
        assert summary["episode_success"] is False and summary["collision_count"] == 4
        assert len(saved["steps"]) == 5 and saved["steps"][1]["phase"] == "policy"

    # Missing APIs and ambiguous tasks must remain visible in the schema.
    domain.parsed_problem = {}
    domain.obj_of_interest = []
    missing = new_episode_metrics(env)
    del domain.sim
    record_step(missing, env, obs, closed_action)
    step = missing["steps"][0]
    assert step["target_unresolved"] and step["target_object_xyz"] is None
    assert step["gripper_object_distance"] is None and step["pick_success"] is None
    assert step["collision_count"] is None and not step["contacts_available"]
    with TemporaryDirectory() as directory:
        summary = save_episode_metrics(missing, Path(directory) / "missing.mp4", False)
        assert summary["collision_count"] is None and summary["pick_success"] is None
        missing["steps"].append(dict(step, pick_success=False, collision_count=0))
        summary = save_episode_metrics(missing, Path(directory) / "partial.mp4", False)
        assert summary["pick_success"] is None and summary["collision_count"] is None

    domain.sim = Fake(model=model, data=data)
    domain.obj_of_interest = ["plate", "bowl"]
    assert new_episode_metrics(env)["target_object"]["ambiguous"]
    del domain.obj_body_id
    assert new_episode_metrics(env)["initial_object_xyz"] == [0.2, 0.0, 0.8]
    domain.parsed_problem = {"goal_state": [["open", "drawer_region"]],
                             "regions": {"drawer_region": {"target": "drawer"}}}
    domain.fixtures_dict = {"drawer": plate}
    assert new_episode_metrics(env)["target_object"]["name"] == "drawer"

    # Pair cap must not cap contact counts; duplicate points count separately.
    contacts(*([(0, 4)] * 25))
    metrics = new_episode_metrics(env)
    record_step(metrics, env, obs, closed_action)
    assert metrics["steps"][0]["collision_count"] == 25
    assert len(metrics["steps"][0]["collision_pairs"]) == 1
    names.extend(f"scene_geom{index}" for index in range(25))
    contacts(*[(0, index) for index in range(6, 31)])
    record_step(metrics, env, obs, closed_action)
    assert metrics["steps"][-1]["collision_count"] == 25
    assert len(metrics["steps"][-1]["collision_pairs"]) == 20
    print("PASS: metrics schema, JSON round-trip, target resolution, lift/support/open/contact checks, API fallbacks")


if __name__ == "__main__":
    selfcheck()
