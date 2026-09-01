"""WebSocket endpoint for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Connected WebSocket clients
_clients: set[WebSocket] = set()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for streaming live trading updates."""
    await websocket.accept()
    _clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(_clients))

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            # Echo acknowledgement
            await websocket.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        _clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_clients))


async def broadcast(event_type: str, data: dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data})
    disconnected = set()

    for client in _clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    _clients.difference_update(disconnected)
