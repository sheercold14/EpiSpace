"""Project CLI; heavy backends are discovered lazily by future plugins."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer

from spatial_episode import __version__
from spatial_episode.application.services.config import load_yaml_config
from spatial_episode.contracts.config_v1 import EpisodeRecipeV1, SensorPresetV1
from spatial_episode.contracts.schema import export_schemas

app = typer.Typer(no_args_is_help=True, help="Spatial Episode Forge")
contracts_app = typer.Typer(no_args_is_help=True, help="Versioned data contracts")
config_app = typer.Typer(no_args_is_help=True, help="Strict semantic configuration")
app.add_typer(contracts_app, name="contracts")
app.add_typer(config_app, name="config")


@app.command()
def doctor() -> None:
    """Report the core environment without importing heavy simulator packages."""

    render_devices = sorted(Path("/dev/dri").glob("renderD*"))
    report = {
        "project_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core": {
            "pydantic": importlib.util.find_spec("pydantic") is not None,
            "numpy": importlib.util.find_spec("numpy") is not None,
        },
        "optional_workers": {
            "habitat_sim": importlib.util.find_spec("habitat_sim") is not None,
            "blender_bpy": importlib.util.find_spec("bpy") is not None,
        },
        "drm_render_devices": [
            {
                "path": str(path),
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
            }
            for path in render_devices
        ],
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@contracts_app.command("export")
def contracts_export(
    output: Annotated[
        Path,
        typer.Option(help="Schema output directory"),
    ] = Path("contracts/jsonschema"),
) -> None:
    """Export deterministic JSON Schemas from the authoritative Python models."""

    for path in export_schemas(output):
        typer.echo(path.as_posix())


@config_app.command("validate")
def config_validate(
    path: Path,
    kind: Annotated[
        str,
        typer.Option(help="Configuration kind: sensor or recipe"),
    ],
) -> None:
    """Validate a YAML config and reject every unknown field."""

    config: SensorPresetV1 | EpisodeRecipeV1
    if kind == "sensor":
        config = load_yaml_config(path, SensorPresetV1)
    elif kind == "recipe":
        config = load_yaml_config(path, EpisodeRecipeV1)
    else:
        raise typer.BadParameter("kind must be sensor or recipe", param_hint="--kind")
    typer.echo(config.model_dump_json(indent=2))
