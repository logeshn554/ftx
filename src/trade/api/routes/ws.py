"""WebSocket endpoint for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from trade.core.secrets import get_ws_token

logger = logging.getLogger(__name__)

router = APIRouter()

# Connected WebSocket clients
_clients: set[WebSocket] = set()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query("")):
    """WebSocket for streaming live trading updates.

    Args:
        token: WebSocket authentication token (via query parameter).
               Must match TRADE_WS_TOKEN environment variable if configured.

    Example:
        ws://localhost:8000/ws?token=YOUR_TOKEN
    """
    # FIX 20: Authenticate WebSocket connection
    ws_token = get_ws_token()
    if ws_token and not secrets.compare_digest(token, ws_token):
        logger.warning("WebSocket connection rejected: invalid token")
        await websocket.close(code=1008, reason="Unauthorized")
        return

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
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        _clients.discard(websocket)


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
