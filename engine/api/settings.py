from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.io import manifests, paths

router = APIRouter()


PROVIDER_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "description": "Gateway for multiple multimodal models.",
        "default_base_url": "https://openrouter.ai/api/v1",
        "models": [
            {"id": "google/gemini-2.0-flash-001", "label": "Gemini 2.0 Flash"},
            {"id": "openai/gpt-4.1-mini", "label": "GPT-4.1 Mini"},
            {"id": "anthropic/claude-3.5-sonnet", "label": "Claude 3.5 Sonnet"}
        ]
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "Native multimodal GPT models.",
        "default_base_url": "https://api.openai.com/v1",
        "models": [
            {"id": "gpt-4.1-mini", "label": "GPT-4.1 Mini"},
            {"id": "gpt-4.1", "label": "GPT-4.1"},
            {"id": "gpt-4o", "label": "GPT-4o"}
        ]
    },
    {
        "id": "groq",
        "name": "Groq",
        "description": "Low-latency inference for open models.",
        "default_base_url": "https://api.groq.com/openai/v1",
        "models": [
            {"id": "llama-3.2-90b-vision-preview", "label": "Llama 3.2 90B Vision"},
            {"id": "llama-3.2-11b-vision-preview", "label": "Llama 3.2 11B Vision"},
            {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "label": "Llama 4 Scout 17B"}
        ]
    },
    {
        "id": "google",
        "name": "Google AI Studio",
        "description": "Gemini family for language and vision.",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": [
            {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash"},
            {"id": "gemini-1.5-pro", "label": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"}
        ]
    }
]

USE_CASES: List[Dict[str, str]] = [
    {
        "id": "preclassification",
        "label": "Pre-classification",
        "description": "Initial class suggestion to speed up annotation."
    },
    {
        "id": "chatbot",
        "label": "Chatbot",
        "description": "Product assistant for questions and training best practices."
    },
    {
        "id": "image_validation",
        "label": "Image validation",
        "description": "Multimodal image review for dataset quality."
    }
]

DEFAULT_ACTIVE_PROVIDER = PROVIDER_CATALOG[0]["id"]
PROVIDER_BY_ID = {provider["id"]: provider for provider in PROVIDER_CATALOG}


class ProviderSettingsInput(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    enabled: bool | None = None


class UseCaseSettingsInput(BaseModel):
    provider: str | None = None
    model: str | None = None


class LlmSettingsInput(BaseModel):
    active_provider: str | None = None
    providers: Dict[str, ProviderSettingsInput] | None = None
    assignments: Dict[str, UseCaseSettingsInput] | None = None


def _first_model(provider_id: str) -> str:
    provider = PROVIDER_BY_ID.get(provider_id) or PROVIDER_BY_ID[DEFAULT_ACTIVE_PROVIDER]
    models = provider.get("models") or []
    if not models:
        return ""
    return models[0]["id"]


def _valid_model(provider_id: str, model_id: str) -> bool:
    provider = PROVIDER_BY_ID.get(provider_id)
    if not provider:
        return False
    return any(model.get("id") == model_id for model in provider.get("models") or [])


def _default_settings() -> Dict[str, Any]:
    providers: Dict[str, Dict[str, Any]] = {}
    for provider in PROVIDER_CATALOG:
        providers[provider["id"]] = {
            "enabled": True,
            "api_key": "",
            "base_url": provider["default_base_url"]
        }

    assignments: Dict[str, Dict[str, Any]] = {}
    for use_case in USE_CASES:
        assignments[use_case["id"]] = {
            "provider": DEFAULT_ACTIVE_PROVIDER,
            "model": _first_model(DEFAULT_ACTIVE_PROVIDER)
        }

    return {
        "active_provider": DEFAULT_ACTIVE_PROVIDER,
        "providers": providers,
        "assignments": assignments
    }


def _normalize_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _default_settings()
    merged = {
        "active_provider": payload.get("active_provider", defaults["active_provider"]),
        "providers": payload.get("providers", defaults["providers"]),
        "assignments": payload.get("assignments", defaults["assignments"])
    }

    active_provider = str(merged.get("active_provider") or DEFAULT_ACTIVE_PROVIDER)
    if active_provider not in PROVIDER_BY_ID:
        active_provider = DEFAULT_ACTIVE_PROVIDER

    providers: Dict[str, Dict[str, Any]] = {}
    incoming_providers = merged.get("providers") or {}
    for provider in PROVIDER_CATALOG:
        provider_id = provider["id"]
        incoming = incoming_providers.get(provider_id) or {}
        api_key = str(incoming.get("api_key") or "")
        base_url = str(incoming.get("base_url") or provider["default_base_url"])
        enabled = bool(incoming.get("enabled", True))
        providers[provider_id] = {
            "enabled": enabled,
            "api_key": api_key,
            "base_url": base_url
        }

    assignments: Dict[str, Dict[str, str]] = {}
    incoming_assignments = merged.get("assignments") or {}
    for use_case in USE_CASES:
        use_case_id = use_case["id"]
        incoming = incoming_assignments.get(use_case_id) or {}
        provider_id = str(incoming.get("provider") or active_provider)
        if provider_id not in PROVIDER_BY_ID:
            provider_id = active_provider
        model_id = str(incoming.get("model") or _first_model(provider_id))
        if not _valid_model(provider_id, model_id):
            model_id = _first_model(provider_id)
        assignments[use_case_id] = {
            "provider": provider_id,
            "model": model_id
        }

    return {
        "active_provider": active_provider,
        "providers": providers,
        "assignments": assignments
    }


def _read_settings() -> Dict[str, Any]:
    settings_path = paths.get_llm_settings_path()
    if not settings_path.exists():
        settings = _default_settings()
        manifests.save_manifest(settings_path, settings)
        return settings
    try:
        raw = manifests.load_manifest(settings_path)
        settings = _normalize_settings(raw)
        manifests.save_manifest(settings_path, settings)
        return settings
    except Exception:
        settings = _default_settings()
        manifests.save_manifest(settings_path, settings)
        return settings


def _save_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    settings = _normalize_settings(payload)
    manifests.save_manifest(paths.get_llm_settings_path(), settings)
    return settings


@router.get("/settings/llm")
async def get_llm_settings():
    settings = _read_settings()
    return {
        "settings": settings,
        "catalog": {
            "providers": PROVIDER_CATALOG,
            "use_cases": USE_CASES
        }
    }


@router.put("/settings/llm")
async def update_llm_settings(request: LlmSettingsInput):
    try:
        current = _read_settings()
        incoming = request.dict(exclude_none=True)
        merged = {
            **current,
            **incoming
        }
        if "providers" in incoming:
            merged["providers"] = {
                **current.get("providers", {}),
                **incoming["providers"]
            }
        if "assignments" in incoming:
            merged["assignments"] = {
                **current.get("assignments", {}),
                **incoming["assignments"]
            }
        settings = _save_settings(merged)
        return {
            "settings": settings,
            "catalog": {
                "providers": PROVIDER_CATALOG,
                "use_cases": USE_CASES
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
