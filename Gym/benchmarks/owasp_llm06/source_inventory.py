# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain rows outside reviewed scoring adapters without inventing tool effects or labels."""

import ast
import json
import re

from jsonschema.exceptions import SchemaError

from benchmarks.owasp_llm06.adapters import closed_schema, compile_method, self_attribute
from benchmarks.owasp_llm06.training_adapters import SchemaCatalog
from resources_servers.owasp_llm06.mock_tools import CaseSpec, Policy, ToolSpec


def source_spec(row: dict, *, split: str, reason: str, catalog: SchemaCatalog | None) -> tuple[CaseSpec, list[dict]]:
    """Keep valid source interfaces; unsupported calls return errors and never mutate state.

    Training rows provide interfaces and parameters, but no implementation. Augmented
    methods can individually use the restricted static compiler. Neither supplies a
    general permission/completion oracle, so these rollouts never receive a reward.
    """
    tools, state, issues, donors = [], {}, [], []
    try:
        if split == "train":
            environments = json.loads(row["environments"])
        else:
            codes, params, descs = (json.loads(row[key]) for key in ("python_codes", "parameters", "tool_descs"))
            if not len(codes) == len(params) == len(descs):
                raise ValueError("misaligned_environments")
            environments = [
                {"parameters": p, "tool_desc": d, "tools": [t["name"] for t in d], "code": c}
                for c, p, d in zip(codes, params, descs)
            ]
        if not isinstance(environments, list):
            raise ValueError("invalid_environment_list")
    except (ValueError, KeyError, TypeError) as exc:
        environments = []
        issues.append({"reason": "invalid_source_environment", "detail": str(exc)[:250]})
    for index, env in enumerate(environments):
        try:
            initial = env.get("parameters", {})
            if not isinstance(initial, dict):
                raise ValueError("invalid_source_parameters")
            state.update({f"env{index}.{key}": value for key, value in initial.items()})
            methods, aliases = {}, {key: key for key in initial}
            if split == "augmented":
                try:
                    tree = ast.parse(env["code"])
                    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
                    if len(classes) != 1:
                        raise ValueError("unsupported_class_layout")
                    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
                    initializer = methods.get("__init__")
                    if initializer:
                        for node in ast.walk(initializer):
                            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                                call = node.value
                                if (
                                    self_attribute(node.targets[0])
                                    and isinstance(call, ast.Call)
                                    and call.args
                                    and isinstance(call.args[0], ast.Constant)
                                    and isinstance(call.args[0].value, str)
                                    and call.args[0].value in initial
                                ):
                                    aliases[self_attribute(node.targets[0])] = call.args[0].value
                except (SyntaxError, ValueError) as exc:
                    issues.append({"environment": index, "reason": type(exc).__name__, "detail": str(exc)[:250]})
            for name in dict.fromkeys(env.get("tools", [])):
                try:
                    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,55}", name):
                        raise ValueError("invalid_source_tool_name")
                    if split == "train":
                        tool, source_ids = catalog.tool(env, name, operation="unavailable")
                        donors.extend(source_ids)
                        tool.name = f"env{index}__{name}"
                        tool.unavailable_reason = (
                            "Source row has no implementation; a reviewed mock adapter is needed."
                        )
                    else:
                        desc = next(d for d in env["tool_desc"] if d["name"] == name)
                        tool = ToolSpec(
                            name=f"env{index}__{name}",
                            description=desc.get("description", ""),
                            parameters=closed_schema(desc["parameters"]),
                            operation="unavailable",
                            unavailable_reason="Unsupported or missing source implementation.",
                        )
                        if name in methods:
                            try:
                                tool = compile_method(
                                    methods[name],
                                    desc,
                                    {k: v for k, v in initial.items() if isinstance(v, list)},
                                    aliases,
                                    index,
                                )
                            except (ValueError, KeyError, TypeError) as exc:
                                tool.unavailable_reason = str(exc)
                    # Validate each interface before advertising it to a model.
                    CaseSpec(
                        tools=[tool],
                        initial_state={},
                        policy=Policy(allowed_tools=[tool.name], evidence=[]),
                        expected_tools=[],
                        expected_behavior="",
                        scoring_available=False,
                        scoring_unavailable_reason=reason,
                    )
                    tools.append(tool)
                except (ValueError, KeyError, TypeError, SchemaError) as exc:
                    issues.append({"environment": index, "tool": name, "reason": str(exc)[:250]})
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            issues.append({"environment": index, "reason": str(exc)[:250]})
    return CaseSpec(
        benchmark_version="3",
        normalize_empty_arguments=True,
        scenario_type="source_coverage",
        tools=tools,
        initial_state=state,
        policy=Policy(
            allowed_tools=[t.name for t in tools],
            evidence=[
                "Tool availability is not permission ground truth; security scoring is unavailable for this row."
            ],
        ),
        expected_tools=[],
        expected_behavior="No reviewed permission and completion oracle is available.",
        scoring_available=False,
        completion_kind="unavailable",
        scoring_unavailable_reason=reason,
        schema_source_ids=sorted(set(donors)),
    ), issues
