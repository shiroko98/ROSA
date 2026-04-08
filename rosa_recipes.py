from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class RosaRecipeSpec:
    name: str
    description: str
    train_mode: str
    memory_mode: str
    backend: str
    seq_address_mode: str
    value_mode: str
    context_gate: bool
    use_match_len_gate: bool
    inject_layers: int
    inject_layer_ids: str
    min_match_len: int
    scale: float


ROSA_RECIPES: Dict[str, RosaRecipeSpec] = {
    "online_v1": RosaRecipeSpec(
        name="online_v1",
        description="在线主线 V1：online_seq + online_sam + shared value + 单早层 + context gate。",
        train_mode="online_seq",
        memory_mode="doc_local",
        backend="sam",
        seq_address_mode="online_sam",
        value_mode="shared",
        context_gate=True,
        use_match_len_gate=True,
        inject_layers=1,
        inject_layer_ids="0",
        min_match_len=1,
        scale=0.15,
    ),
    "online_v2": RosaRecipeSpec(
        name="online_v2",
        description="在线主线 V2：online_seq + online_sam + per-layer value + 单早层 + context gate。",
        train_mode="online_seq",
        memory_mode="doc_local",
        backend="sam",
        seq_address_mode="online_sam",
        value_mode="per_layer",
        context_gate=True,
        use_match_len_gate=True,
        inject_layers=1,
        inject_layer_ids="0",
        min_match_len=1,
        scale=0.15,
    ),
}


def available_rosa_recipe_names() -> List[str]:
    return ["custom"] + sorted(ROSA_RECIPES.keys())


def get_rosa_recipe_spec(name: str) -> RosaRecipeSpec | None:
    return ROSA_RECIPES.get(name)


def recipe_fields(spec: RosaRecipeSpec) -> Dict[str, Any]:
    return {
        "rosa_train_mode": spec.train_mode,
        "rosa_memory_mode": spec.memory_mode,
        "rosa_backend": spec.backend,
        "rosa_seq_address_mode": spec.seq_address_mode,
        "rosa_value_mode": spec.value_mode,
        "rosa_context_gate": spec.context_gate,
        "rosa_disable_match_len_gate": not spec.use_match_len_gate,
        "rosa_inject_layers": spec.inject_layers,
        "rosa_inject_layer_ids": spec.inject_layer_ids,
        "rosa_min_match_len": spec.min_match_len,
        "rosa_scale": spec.scale,
    }


def apply_rosa_recipe(args) -> Dict[str, Any]:
    recipe_name = getattr(args, "rosa_recipe", "custom")
    if recipe_name == "custom":
        return {
            "name": "custom",
            "description": "不应用预设，由命令行参数直接控制。",
            "applied": False,
            "applied_fields": {},
        }

    spec = get_rosa_recipe_spec(recipe_name)
    if spec is None:
        raise ValueError(f"未知 ROSA recipe: {recipe_name}")

    applied_fields = recipe_fields(spec)
    for key, value in applied_fields.items():
        setattr(args, key, value)

    return {
        "name": spec.name,
        "description": spec.description,
        "applied": True,
        "applied_fields": applied_fields,
        "spec": asdict(spec),
    }
