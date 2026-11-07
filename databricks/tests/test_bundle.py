"""Static checks of the bundle YAML, so mistakes show up before `databricks bundle validate`.

Set DW_BUNDLE_SCHEMA to the output of `databricks bundle schema` to also validate every
YAML file against the official bundle JSON schema (CI does this).
"""
import json
import os
import re
import tomllib
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]           # databricks/
ADDONS = [HERE / "addons" / "serving" / "databricks.yml", HERE / "addons" / "monitor" / "databricks.yml"]
FILES = [HERE / "databricks.yml", *sorted((HERE / "resources").glob("*.yml")), *ADDONS]


def load(path):
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def merged():
    """databricks.yml plus every included resource file, merged at resources level."""
    root = load(HERE / "databricks.yml")
    resources = {}
    for f in sorted((HERE / "resources").glob("*.yml")):
        for kind, items in (load(f).get("resources") or {}).items():
            resources.setdefault(kind, {}).update(items)
    root["resources"] = resources
    return root


def _walk(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path + (k,), v
            yield from _walk(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,))


def test_serverless_only_no_cluster_definitions():
    for f in FILES:
        for path, _ in _walk(load(f)):
            assert path[-1] not in ("new_cluster", "job_clusters", "clusters", "existing_cluster_id",
                                    "node_type_id", "instance_pool_id"), f"{f.name}: {path}"


def test_resource_keys_are_unique_across_types(merged):
    keys = [k for items in merged["resources"].values() for k in items]
    assert len(keys) == len(set(keys))


def test_job_dag_environments_and_entry_points(merged):
    job = merged["resources"]["jobs"]["fairness_audit"]
    tasks = {t["task_key"]: t for t in job["tasks"]}
    envs = {e["environment_key"] for e in job["environments"]}
    scripts = tomllib.loads((HERE / "pyproject.toml").read_text())["project"]["scripts"]
    from driftwatch_databricks import tasks as task_module
    for t in tasks.values():
        for dep in t.get("depends_on", []):
            assert dep["task_key"] in tasks
        if "python_wheel_task" in t:
            assert t["environment_key"] in envs
            w = t["python_wheel_task"]
            assert w["package_name"] == "driftwatch_databricks"
            target = scripts[w["entry_point"]].split(":")[1]
            assert hasattr(task_module, target)
    assert tasks["train_model"]["depends_on"] == [{"task_key": "should_retrain", "outcome": "true"}]
    assert tasks["run_audit"]["run_if"] == "NONE_FAILED"
    # Free Edition runs at most 5 tasks at once; this DAG never runs more than 1 in parallel.
    assert len(tasks) <= 5
    assert job["schedule"]["pause_status"] == "${var.schedule_pause_status}"
    assert merged["variables"]["schedule_pause_status"]["default"] == "PAUSED"


def test_job_parameters_are_used_and_named_parameters_are_cli_flags(merged):
    job = merged["resources"]["jobs"]["fairness_audit"]
    text = json.dumps(job["tasks"])
    for p in job["parameters"]:
        assert "{{job.parameters.%s}}" % p["name"] in text, p["name"]
    # named_parameters reach the entry point as --key=value: each key must be a known flag.
    src = (HERE / "src" / "driftwatch_databricks" / "tasks.py").read_text()
    for t in job["tasks"]:
        for key in t.get("python_wheel_task", {}).get("named_parameters", {}):
            assert f'"--{key}"' in src, f"{t['task_key']}: no --{key} argument"


def test_local_paths_exist(merged):
    res = merged["resources"]
    pipe = res["pipelines"]["medallion"]
    base = HERE / "resources"
    assert (base / pipe["root_path"] / "driftwatch_databricks" / "__init__.py").exists()
    for lib in pipe["libraries"]:
        assert (base / lib["file"]["path"]).exists()
    app = res["apps"]["driftwatch_app"]
    for name in ("app.py", "app.yaml", "requirements.txt", "queries.py"):
        assert (base / app["source_code_path"] / name).exists()
    for art in merged["artifacts"].values():
        assert (HERE / art["path"] / "pyproject.toml").exists()


def test_pipeline_is_serverless_and_uses_the_shared_rules(merged):
    pipe = merged["resources"]["pipelines"]["medallion"]
    assert pipe["serverless"] is True and "continuous" not in pipe
    src = (HERE / "src" / "pipeline" / "driftwatch_pipeline.py").read_text()
    assert "expect_all_or_drop(DROP_RULES)" in src and "expect_all(WARN_RULES)" in src


def test_app_env_matches_app_yaml():
    app = load(HERE / "resources" / "app.yml")["resources"]["apps"]["driftwatch_app"]
    file_env = {e["name"] for e in load(HERE / "app" / "app.yaml")["env"]}
    bundle_env = {e["name"] for e in app["config"]["env"]}
    assert file_env == bundle_env
    assert app["config"]["command"] == load(HERE / "app" / "app.yaml")["command"]


def test_addons_point_at_the_main_bundle_names():
    """The add-ons must use the same names as the main bundle, including the dev prefix."""
    main = load(HERE / "databricks.yml")
    model = load(HERE / "resources" / "unity_catalog.yml")["resources"]["registered_models"]["readmission_model"]
    for f in ADDONS:
        doc = load(f)
        assert doc["variables"]["catalog"]["default"] == main["variables"]["catalog"]["default"]
        assert doc["variables"]["schema"]["default"] == main["variables"]["schema"]["default"]
        assert doc["targets"]["dev"]["variables"]["schema"] == "dev_${workspace.current_user.short_name}_driftwatch"
    serving = load(ADDONS[0])
    assert serving["variables"]["model_name"]["default"] == model["name"]
    entity = serving["resources"]["model_serving_endpoints"]["readmission"]["config"]["served_entities"][0]
    assert entity["workload_type"] == "CPU" and entity["scale_to_zero_enabled"] is True
    monitor = load(ADDONS[1])["resources"]["quality_monitors"]["gold_fairness_metrics"]
    assert monitor["table_name"].endswith(".gold_fairness_metrics")


def _json_safe_schema(schema: dict) -> dict:
    """The bundle schema uses \\p{L} / \\p{N} in regex patterns; Python's re needs another spelling."""
    text = json.dumps(schema)
    text = text.replace("[\\\\p{L}\\\\p{N}]", "[^\\\\W_]")          # letter or digit
    text = text.replace("\\\\p{L}", "[^\\\\W\\\\d_]").replace("\\\\p{N}", "\\\\d")
    return json.loads(text)


@pytest.mark.skipif(not os.getenv("DW_BUNDLE_SCHEMA"), reason="set DW_BUNDLE_SCHEMA to `databricks bundle schema` output")
@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(HERE)))
def test_files_match_official_bundle_schema(path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = _json_safe_schema(json.loads(Path(os.environ["DW_BUNDLE_SCHEMA"]).read_text()))
    doc = load(path)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, "\n".join(f"{list(e.path)}: {e.message[:300]}" for e in errors[:10])


def test_no_secrets_in_files():
    pattern = re.compile(r"(dapi[0-9a-f]{20,}|sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_\-]{20,})")
    for f in HERE.rglob("*"):
        if f.is_file() and f.suffix in (".yml", ".yaml", ".py", ".json", ".toml", ".md", ".txt"):
            assert not pattern.search(f.read_text(errors="ignore")), f
