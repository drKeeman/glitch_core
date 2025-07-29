"""
File storage utilities for data export/import and structured logging.
"""

import json
import gzip
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from pydantic import BaseModel

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class FileStorage:
    """File storage utilities for data management."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize file storage with base path."""
        self.base_path = Path(base_path or settings.DATA_DIR)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.results_path = self.base_path / "results"
        
        for path in [self.raw_path, self.processed_path, self.results_path]:
            path.mkdir(exist_ok=True)
    
    def save_json(self, data: Any, filename: str, compress: bool = False) -> bool:
        """Save data as JSON file."""
        try:
            file_path = self.base_path / filename
            
            # Convert Pydantic models to dict
            if isinstance(data, BaseModel):
                data = data.model_dump()
            elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
                data = [item.model_dump() for item in data]
            
            # Save with optional compression
            if compress:
                file_path = file_path.with_suffix('.json.gz')
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info("Data saved to JSON", filename=filename, compressed=compress)
            return True
            
        except Exception as e:
            logger.error("Failed to save JSON", filename=filename, error=str(e))
            return False
    
    def load_json(self, filename: str, compressed: bool = False) -> Optional[Any]:
        """Load data from JSON file."""
        try:
            file_path = self.base_path / filename
            
            if compressed:
                file_path = file_path.with_suffix('.json.gz')
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            logger.info("Data loaded from JSON", filename=filename)
            return data
            
        except Exception as e:
            logger.error("Failed to load JSON", filename=filename, error=str(e))
            return None
    
    def save_pickle(self, data: Any, filename: str, compress: bool = False) -> bool:
        """Save data as pickle file."""
        try:
            file_path = self.base_path / filename
            
            if compress:
                file_path = file_path.with_suffix('.pkl.gz')
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info("Data saved to pickle", filename=filename, compressed=compress)
            return True
            
        except Exception as e:
            logger.error("Failed to save pickle", filename=filename, error=str(e))
            return False
    
    def load_pickle(self, filename: str, compressed: bool = False) -> Optional[Any]:
        """Load data from pickle file."""
        try:
            file_path = self.base_path / filename
            
            if compressed:
                file_path = file_path.with_suffix('.pkl.gz')
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.info("Data loaded from pickle", filename=filename)
            return data
            
        except Exception as e:
            logger.error("Failed to load pickle", filename=filename, error=str(e))
            return None
    
    def save_simulation_data(self, simulation_id: str, data: Dict[str, Any], data_type: str) -> bool:
        """Save simulation data with organized structure."""
        try:
            # Create simulation-specific directory
            sim_path = self.raw_path / simulation_id
            sim_path.mkdir(exist_ok=True)
            
            # Create type-specific subdirectory
            type_path = sim_path / data_type
            type_path.mkdir(exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}.json"
            
            file_path = type_path / filename
            
            # Save data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Simulation data saved", simulation_id=simulation_id, data_type=data_type, filename=filename)
            return True
            
        except Exception as e:
            logger.error("Failed to save simulation data", simulation_id=simulation_id, data_type=data_type, error=str(e))
            return False
    
    def load_simulation_data(self, simulation_id: str, data_type: str, filename: Optional[str] = None) -> Optional[Any]:
        """Load simulation data."""
        try:
            sim_path = self.raw_path / simulation_id / data_type
            
            if filename:
                file_path = sim_path / filename
            else:
                # Load most recent file
                files = list(sim_path.glob("*.json"))
                if not files:
                    return None
                file_path = max(files, key=lambda f: f.stat().st_mtime)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info("Simulation data loaded", simulation_id=simulation_id, data_type=data_type, filename=file_path.name)
            return data
            
        except Exception as e:
            logger.error("Failed to load simulation data", simulation_id=simulation_id, data_type=data_type, error=str(e))
            return None
    
    def save_persona_data(self, persona_id: str, data: Dict[str, Any]) -> bool:
        """Save persona-specific data."""
        try:
            persona_path = self.raw_path / "personas" / persona_id
            persona_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"persona_{timestamp}.json"
            
            file_path = persona_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Persona data saved", persona_id=persona_id, filename=filename)
            return True
            
        except Exception as e:
            logger.error("Failed to save persona data", persona_id=persona_id, error=str(e))
            return False
    
    def save_assessment_data(self, assessment_id: str, data: Dict[str, Any]) -> bool:
        """Save assessment data."""
        try:
            assessment_path = self.raw_path / "assessments"
            assessment_path.mkdir(exist_ok=True)
            
            filename = f"assessment_{assessment_id}.json"
            file_path = assessment_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Assessment data saved", assessment_id=assessment_id)
            return True
            
        except Exception as e:
            logger.error("Failed to save assessment data", assessment_id=assessment_id, error=str(e))
            return False
    
    def save_mechanistic_data(self, analysis_id: str, data: Dict[str, Any]) -> bool:
        """Save mechanistic analysis data."""
        try:
            mechanistic_path = self.raw_path / "mechanistic"
            mechanistic_path.mkdir(exist_ok=True)
            
            filename = f"analysis_{analysis_id}.json"
            file_path = mechanistic_path / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Mechanistic data saved", analysis_id=analysis_id)
            return True
            
        except Exception as e:
            logger.error("Failed to save mechanistic data", analysis_id=analysis_id, error=str(e))
            return False
    
    def export_results(self, simulation_id: str, format: str = "json") -> bool:
        """Export simulation results."""
        try:
            results_path = self.results_path / simulation_id
            results_path.mkdir(exist_ok=True)
            
            # Collect all data for export
            export_data = {
                "simulation_id": simulation_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "data_sources": []
            }
            
            # Export different data types
            data_types = ["personas", "assessments", "events", "mechanistic"]
            
            for data_type in data_types:
                data_path = self.raw_path / simulation_id / data_type
                if data_path.exists():
                    files = list(data_path.glob("*.json"))
                    if files:
                        export_data["data_sources"].append({
                            "type": data_type,
                            "file_count": len(files),
                            "files": [f.name for f in files]
                        })
            
            # Save export manifest
            manifest_file = results_path / f"export_manifest_{simulation_id}.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info("Results exported", simulation_id=simulation_id, format=format)
            return True
            
        except Exception as e:
            logger.error("Failed to export results", simulation_id=simulation_id, error=str(e))
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old data files."""
        try:
            cutoff_time = datetime.utcnow().timestamp() - (days_to_keep * 24 * 3600)
            deleted_count = 0
            
            for file_path in self.base_path.rglob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info("Old data cleaned up", deleted_count=deleted_count, days_kept=days_to_keep)
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                "total_size_mb": 0,
                "file_count": 0,
                "directory_count": 0,
                "by_type": {}
            }
            
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    stats["file_count"] += 1
                    stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                    
                    # Count by file type
                    file_type = file_path.suffix
                    if file_type not in stats["by_type"]:
                        stats["by_type"][file_type] = {"count": 0, "size_mb": 0}
                    stats["by_type"][file_type]["count"] += 1
                    stats["by_type"][file_type]["size_mb"] += file_path.stat().st_size / (1024 * 1024)
                elif file_path.is_dir():
                    stats["directory_count"] += 1
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get storage stats", error=str(e))
            return {}


# Global file storage instance
file_storage = FileStorage() 