"""
WebSocket endpoints for real-time monitoring.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger
from src.services.simulation_engine import SimulationEngine

logger = get_logger(__name__)
router = APIRouter(tags=["websocket"])


class WebSocketMessageType(str, Enum):
    """WebSocket message types."""
    SIMULATION_STATUS = "simulation_status"
    PROGRESS_UPDATE = "progress_update"
    EVENT_OCCURRED = "event_occurred"
    ASSESSMENT_COMPLETED = "assessment_completed"
    MECHANISTIC_UPDATE = "mechanistic_update"
    ERROR = "error"
    ALERT = "alert"


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: WebSocketMessageType
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self, simulation_engine: SimulationEngine = None):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
        self.simulation_engine = simulation_engine or SimulationEngine()
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        message_json = message.model_dump_json()
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_simulation_status(self, status_data: Dict[str, Any]):
        """Broadcast simulation status update."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SIMULATION_STATUS,
            data=status_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_progress_update(self, progress_data: Dict[str, Any]):
        """Broadcast progress update."""
        message = WebSocketMessage(
            type=WebSocketMessageType.PROGRESS_UPDATE,
            data=progress_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_event_occurred(self, event_data: Dict[str, Any]):
        """Broadcast event occurrence."""
        message = WebSocketMessage(
            type=WebSocketMessageType.EVENT_OCCURRED,
            data=event_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_assessment_completed(self, assessment_data: Dict[str, Any]):
        """Broadcast assessment completion."""
        message = WebSocketMessage(
            type=WebSocketMessageType.ASSESSMENT_COMPLETED,
            data=assessment_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_mechanistic_update(self, mechanistic_data: Dict[str, Any]):
        """Broadcast mechanistic analysis update."""
        message = WebSocketMessage(
            type=WebSocketMessageType.MECHANISTIC_UPDATE,
            data=mechanistic_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_error(self, error_data: Dict[str, Any]):
        """Broadcast error message."""
        message = WebSocketMessage(
            type=WebSocketMessageType.ERROR,
            data=error_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert message."""
        message = WebSocketMessage(
            type=WebSocketMessageType.ALERT,
            data=alert_data,
            timestamp=datetime.now(timezone.utc)
        )
        await self.broadcast(message)


# Global WebSocket manager instance
from src.api.routes.simulation import simulation_engine
websocket_manager = WebSocketManager(simulation_engine)


@router.websocket("/ws/simulation")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """WebSocket endpoint for simulation monitoring."""
    await websocket_manager.connect(websocket)
    
    try:
        # Send initial status
        status_data = await websocket_manager.simulation_engine.get_simulation_status()
        logger.info(f"Initial status data: {status_data}")
        if status_data:
            await websocket_manager.send_personal_message(
                WebSocketMessage(
                    type=WebSocketMessageType.SIMULATION_STATUS,
                    data=status_data,
                    timestamp=datetime.now(timezone.utc)
                ).model_dump_json(),
                websocket
            )
        else:
            logger.info("No simulation status available for initial message")
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                if message_data.get("type") == "request_status":
                    status = await websocket_manager.simulation_engine.get_simulation_status()
                    await websocket_manager.send_personal_message(
                        WebSocketMessage(
                            type=WebSocketMessageType.SIMULATION_STATUS,
                            data=status or {},
                            timestamp=datetime.now(timezone.utc)
                        ).model_dump_json(),
                        websocket
                    )
                
                elif message_data.get("type") == "request_progress":
                    # Send mock progress data
                    progress_data = {
                        "current_day": 5,
                        "total_days": 30,
                        "progress_percentage": 16.67,
                        "active_personas": 9,
                        "events_processed": 25,
                        "assessments_completed": 15
                    }
                    await websocket_manager.send_personal_message(
                        WebSocketMessage(
                            type=WebSocketMessageType.PROGRESS_UPDATE,
                            data=progress_data,
                            timestamp=datetime.now(timezone.utc)
                        ).model_dump_json(),
                        websocket
                    )
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket_manager.send_personal_message(
                    WebSocketMessage(
                        type=WebSocketMessageType.ERROR,
                        data={"error": str(e)},
                        timestamp=datetime.now(timezone.utc)
                    ).model_dump_json(),
                    websocket
                )
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        websocket_manager.disconnect(websocket)


@router.websocket("/ws/monitoring")
async def websocket_monitoring_endpoint(websocket: WebSocket):
    """WebSocket endpoint for general monitoring."""
    await websocket_manager.connect(websocket)
    
    try:
        # Send initial monitoring data
        monitoring_data = {
            "system_status": "healthy",
            "active_connections": len(websocket_manager.active_connections),
            "simulation_running": False,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket_manager.send_personal_message(
            WebSocketMessage(
                type=WebSocketMessageType.SIMULATION_STATUS,
                data=monitoring_data,
                timestamp=datetime.now(timezone.utc)
            ).model_dump_json(),
            websocket
        )
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Handle monitoring requests
                message_data = json.loads(data)
                
                if message_data.get("type") == "ping":
                    await websocket_manager.send_personal_message(
                        WebSocketMessage(
                            type=WebSocketMessageType.SIMULATION_STATUS,
                            data={"pong": True},
                            timestamp=datetime.now(timezone.utc)
                        ).model_dump_json(),
                        websocket
                    )
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in monitoring WebSocket: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info("Monitoring WebSocket disconnected")
    finally:
        websocket_manager.disconnect(websocket)


# Background task for periodic status updates
async def periodic_status_updates():
    """Send periodic status updates to all connected WebSockets."""
    while True:
        try:
            if websocket_manager.active_connections:
                # Get current simulation status
                status_data = await websocket_manager.simulation_engine.get_simulation_status()
                
                if status_data:
                    await websocket_manager.broadcast_simulation_status(status_data)
                    
                    # Also send as progress update for frontend compatibility
                    progress_data = {
                        "current_day": status_data.get("current_day", 0),
                        "total_days": status_data.get("total_days", 30),
                        "progress_percentage": status_data.get("progress_percentage", 0.0),
                        "active_personas": status_data.get("active_personas", 0),
                        "events_processed": status_data.get("events_processed", 0),
                        "assessments_completed": status_data.get("assessments_completed", 0),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    await websocket_manager.broadcast_progress_update(progress_data)
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in periodic status updates: {e}")
            await asyncio.sleep(10)  # Wait longer on error 