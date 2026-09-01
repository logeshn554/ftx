"""Model registry REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_models():
    """List all registered models."""
    return {"models": []}


@router.get("/production")
async def production_model():
    """Get current production model info."""
    return {"model": None, "message": "No production model registered"}


@router.post("/{version}/promote")
async def promote_model(version: str):
    """Manually promote a model version."""
    return {"promoted": version, "status": "promoted"}


@router.post("/rollback")
async def rollback_model():
    """Rollback to previous production model."""
    return {"status": "rolled_back", "to_version": None}
