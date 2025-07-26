"""
WebSocket endpoints for real-time simulation updates.
"""

import json
import asyncio
from typing import Dict, Set, Any
from uuid import UUID
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

from .models import WebSocketEvent
from glitch_core.config.logging import get_logger

logger = get_logger("websocket_api")


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, experiment_id: str):
        """Connect a WebSocket to an experiment."""
        await websocket.accept()
        
        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()
        
        self.active_connections[experiment_id].add(websocket)
        logger.info("websocket_connected", experiment_id=experiment_id)
    
    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """Disconnect a WebSocket from an experiment."""
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]
        
        logger.info("websocket_disconnected", experiment_id=experiment_id)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("websocket_send_failed", error=str(e))
    
    async def broadcast_to_experiment(self, message: str, experiment_id: str):
        """Broadcast a message to all connections for an experiment."""
        if experiment_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[experiment_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error("websocket_broadcast_failed", error=str(e))
                    disconnected.add(connection)
            
            # Remove disconnected connections
            for connection in disconnected:
                self.active_connections[experiment_id].discard(connection)
            
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for experiment real-time updates."""
    try:
        await manager.connect(websocket, experiment_id)
        
        # Send initial connection confirmation
        initial_event = WebSocketEvent(
            event_type="connection_established",
            experiment_id=UUID(experiment_id),
            timestamp=datetime.utcnow(),
            data={"message": "Connected to experiment updates", "experiment_id": experiment_id}
        )
        
        await manager.send_personal_message(
            initial_event.model_dump_json(), 
            websocket
        )
        
        # Keep connection alive and handle incoming messages
        try:
            while True:
                # Wait for messages from client (ping/pong for keepalive)
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        # Respond with pong
                        pong_event = WebSocketEvent(
                            event_type="pong",
                            experiment_id=UUID(experiment_id),
                            timestamp=datetime.utcnow(),
                            data={"timestamp": datetime.utcnow().isoformat()}
                        )
                        await manager.send_personal_message(
                            pong_event.model_dump_json(),
                            websocket
                        )
                        
                except json.JSONDecodeError:
                    logger.warning("invalid_websocket_message", experiment_id=experiment_id)
                    
        except WebSocketDisconnect:
            logger.info("websocket_disconnect", experiment_id=experiment_id)
            
    except Exception as e:
        logger.error("websocket_error", experiment_id=experiment_id, error=str(e))
    finally:
        manager.disconnect(websocket, experiment_id)


async def send_epoch_update(experiment_id: str, epoch: int, data: Dict[str, Any]):
    """Send epoch completion update to all connected clients."""
    event = WebSocketEvent(
        event_type="epoch_completed",
        experiment_id=UUID(experiment_id),
        timestamp=datetime.utcnow(),
        data={
            "epoch": epoch,
            **data
        }
    )
    
    await manager.broadcast_to_experiment(
        event.model_dump_json(),
        experiment_id
    )
    
    logger.info("epoch_update_sent", experiment_id=experiment_id, epoch=epoch)


async def send_pattern_emergence(experiment_id: str, pattern_data: Dict[str, Any]):
    """Send pattern emergence notification to all connected clients."""
    event = WebSocketEvent(
        event_type="pattern_emerged",
        experiment_id=UUID(experiment_id),
        timestamp=datetime.utcnow(),
        data=pattern_data
    )
    
    await manager.broadcast_to_experiment(
        event.model_dump_json(),
        experiment_id
    )
    
    logger.info("pattern_emergence_sent", experiment_id=experiment_id, pattern_type=pattern_data.get("pattern_type"))


async def send_stability_warning(experiment_id: str, warning_data: Dict[str, Any]):
    """Send stability warning to all connected clients."""
    event = WebSocketEvent(
        event_type="stability_warning",
        experiment_id=UUID(experiment_id),
        timestamp=datetime.utcnow(),
        data=warning_data
    )
    
    await manager.broadcast_to_experiment(
        event.model_dump_json(),
        experiment_id
    )
    
    logger.info("stability_warning_sent", experiment_id=experiment_id, risk_level=warning_data.get("risk_level"))


async def send_intervention_applied(experiment_id: str, intervention_data: Dict[str, Any]):
    """Send intervention application notification to all connected clients."""
    event = WebSocketEvent(
        event_type="intervention_applied",
        experiment_id=UUID(experiment_id),
        timestamp=datetime.utcnow(),
        data=intervention_data
    )
    
    await manager.broadcast_to_experiment(
        event.model_dump_json(),
        experiment_id
    )
    
    logger.info("intervention_applied_sent", experiment_id=experiment_id, event_type=intervention_data.get("event_type"))


async def send_simulation_complete(experiment_id: str, completion_data: Dict[str, Any]):
    """Send simulation completion notification to all connected clients."""
    event = WebSocketEvent(
        event_type="simulation_complete",
        experiment_id=UUID(experiment_id),
        timestamp=datetime.utcnow(),
        data=completion_data
    )
    
    await manager.broadcast_to_experiment(
        event.model_dump_json(),
        experiment_id
    )
    
    logger.info("simulation_complete_sent", experiment_id=experiment_id)


# HTML page for testing WebSocket connections
def get_websocket_test_page(experiment_id: str) -> str:
    """Generate HTML page for testing WebSocket connections."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Glitch Core WebSocket Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            #messages {{ height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin: 10px 0; }}
            .message {{ margin: 5px 0; padding: 5px; background: #f0f0f0; }}
            .error {{ background: #ffebee; color: #c62828; }}
            .success {{ background: #e8f5e8; color: #2e7d32; }}
        </style>
    </head>
    <body>
        <h1>Glitch Core WebSocket Test</h1>
        <p>Experiment ID: <strong>{experiment_id}</strong></p>
        <div>
            <button onclick="connect()">Connect</button>
            <button onclick="disconnect()">Disconnect</button>
            <button onclick="sendPing()">Send Ping</button>
        </div>
        <div id="messages"></div>
        
        <script>
            let ws = null;
            
            function connect() {{
                ws = new WebSocket(`ws://localhost:8000/ws/experiments/{experiment_id}`);
                
                ws.onopen = function(event) {{
                    addMessage("Connected to WebSocket", "success");
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    addMessage(`Received: ${{data.event_type}} - ${{JSON.stringify(data.data)}}`, "message");
                }};
                
                ws.onclose = function(event) {{
                    addMessage("Disconnected from WebSocket", "error");
                }};
                
                ws.onerror = function(error) {{
                    addMessage("WebSocket error: " + error, "error");
                }};
            }}
            
            function disconnect() {{
                if (ws) {{
                    ws.close();
                    ws = null;
                }}
            }}
            
            function sendPing() {{
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{"type": "ping"}}));
                    addMessage("Sent ping", "message");
                }}
            }}
            
            function addMessage(message, type) {{
                const messagesDiv = document.getElementById("messages");
                const messageDiv = document.createElement("div");
                messageDiv.className = `message ${{type}}`;
                messageDiv.textContent = `[${{new Date().toLocaleTimeString()}}] ${{message}}`;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}
        </script>
    </body>
    </html>
    """ 