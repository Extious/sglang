from __future__ import annotations

from typing import Any

from benchmark.fault_tolerance.simulator.synthesis_requests.common.schema import (
    SyntheticRequest,
)


def with_total_request(
    request: SyntheticRequest,
    *,
    total_request: int,
) -> dict[str, Any]:
    simulation = dict(request.custom_params)
    simulation["total_request"] = int(total_request)
    return simulation


def _build_payload(
    request: SyntheticRequest,
    *,
    total_request: int,
) -> dict[str, Any]:
    return {
        "prompt": "",
        "input_ids": list(request.token_ids),
        "sampling_params": {
            "ignore_eos": True,
            "max_new_tokens": int(request.output_length),
            "custom_params": {
                "simulation": with_total_request(
                    request,
                    total_request=total_request,
                )
            },
        },
        "routed_dp_rank": int(request.assigned_dp_rank),
    }


def build_generate_kwargs(
    request: SyntheticRequest,
    *,
    total_request: int,
) -> dict[str, Any]:
    return _build_payload(request, total_request=total_request)


def build_http_generate_payload(
    request: SyntheticRequest,
    *,
    total_request: int,
) -> dict[str, Any]:
    return _build_payload(request, total_request=total_request)
