"""
Data export utilities for AI personality drift research.
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

# Use absolute imports instead of relative imports
try:
    from models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from models.persona import PersonalityTrait
except ImportError:
    # Fallback for when running from notebooks
    from src.models import Persona, AssessmentResult, PHQ9Result, GAD7Result, PSS10Result
    from src.models.persona import PersonalityTrait

logger = logging.getLogger(__name__)


class DataExporter:
    """Data export utilities for research data."""
    
    def __init__(self, output_dir: str = "data/exports"):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Output directory for exported data
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_assessment_data(self, assessment_results: List[AssessmentResult],
                             format: str = "csv") -> str:
        """
        Export assessment data to various formats.
        
        Args:
            assessment_results: List of assessment results
            format: Export format ("csv", "json", "parquet", "excel")
            
        Returns:
            Path to exported file
        """
        # Convert to DataFrame
        data = []
        for result in assessment_results:
            row = {
                "assessment_id": result.assessment_id,
                "persona_id": result.persona_id,
                "assessment_type": result.assessment_type,
                "simulation_day": result.simulation_day,
                "total_score": result.total_score,
                "severity_level": result.severity_level.value,
                "response_consistency": result.response_consistency,
                "response_time_avg": result.response_time_avg,
                "created_at": result.created_at.isoformat(),
                "raw_responses": json.dumps(result.raw_responses),
                "parsed_scores": json.dumps(result.parsed_scores)
            }
            
            # Add assessment-specific fields
            if isinstance(result, PHQ9Result):
                row.update({
                    "depression_severity": result.depression_severity.value,
                    "suicidal_ideation_score": result.suicidal_ideation_score
                })
            elif isinstance(result, GAD7Result):
                row.update({
                    "anxiety_severity": result.anxiety_severity.value,
                    "worry_duration": result.worry_duration
                })
            elif isinstance(result, PSS10Result):
                row.update({
                    "stress_severity": result.stress_severity.value,
                    "coping_effectiveness": result.coping_effectiveness
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assessment_data_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "excel":
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        self.logger.info(f"Exported assessment data to {filepath}")
        return str(filepath)
    
    def export_persona_data(self, personas: List[Persona],
                           format: str = "csv") -> str:
        """
        Export persona data to various formats.
        
        Args:
            personas: List of personas
            format: Export format ("csv", "json", "parquet", "excel")
            
        Returns:
            Path to exported file
        """
        # Convert to DataFrame
        data = []
        for persona in personas:
            # Baseline data
            baseline_row = {
                "persona_id": persona.state.persona_id,
                "name": persona.baseline.name,
                "age": persona.baseline.age,
                "occupation": persona.baseline.occupation,
                "openness": persona.baseline.openness,
                "conscientiousness": persona.baseline.conscientiousness,
                "extraversion": persona.baseline.extraversion,
                "agreeableness": persona.baseline.agreeableness,
                "neuroticism": persona.baseline.neuroticism,
                "baseline_phq9": persona.baseline.baseline_phq9,
                "baseline_gad7": persona.baseline.baseline_gad7,
                "baseline_pss10": persona.baseline.baseline_pss10,
                "response_length": persona.baseline.response_length,
                "communication_style": persona.baseline.communication_style,
                "emotional_expression": persona.baseline.emotional_expression,
                "core_memories": json.dumps(persona.baseline.core_memories),
                "relationships": json.dumps(persona.baseline.relationships),
                "values": json.dumps(persona.baseline.values),
                "created_at": persona.created_at.isoformat(),
                "version": persona.version
            }
            
            # State data
            state_row = {
                "persona_id": persona.state.persona_id,
                "simulation_day": persona.state.simulation_day,
                "last_assessment_day": persona.state.last_assessment_day,
                "current_phq9": persona.state.current_phq9,
                "current_gad7": persona.state.current_gad7,
                "current_pss10": persona.state.current_pss10,
                "drift_magnitude": persona.state.drift_magnitude,
                "emotional_state": persona.state.emotional_state,
                "stress_level": persona.state.stress_level,
                "recent_events": json.dumps(persona.state.recent_events),
                "trait_changes": json.dumps(persona.state.trait_changes),
                "attention_patterns": json.dumps(persona.state.attention_patterns),
                "activation_changes": json.dumps(persona.state.activation_changes),
                "state_created_at": persona.state.created_at.isoformat(),
                "state_updated_at": persona.state.updated_at.isoformat()
            }
            
            # Combine baseline and state data
            row = {**baseline_row, **state_row}
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"persona_data_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(filepath, index=False)
        elif format == "excel":
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        self.logger.info(f"Exported persona data to {filepath}")
        return str(filepath)
    
    def export_mechanistic_data(self, mechanistic_data: pd.DataFrame,
                               format: str = "csv") -> str:
        """
        Export mechanistic analysis data.
        
        Args:
            mechanistic_data: DataFrame with mechanistic data
            format: Export format ("csv", "json", "parquet", "excel")
            
        Returns:
            Path to exported file
        """
        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mechanistic_data_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "csv":
            mechanistic_data.to_csv(filepath, index=False)
        elif format == "json":
            mechanistic_data.to_json(filepath, orient="records", indent=2)
        elif format == "parquet":
            mechanistic_data.to_parquet(filepath, index=False)
        elif format == "excel":
            mechanistic_data.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        self.logger.info(f"Exported mechanistic data to {filepath}")
        return str(filepath)
    
    def export_simulation_metadata(self, simulation_config: Dict[str, Any],
                                 simulation_results: Dict[str, Any],
                                 format: str = "json") -> str:
        """
        Export simulation metadata and results.
        
        Args:
            simulation_config: Simulation configuration
            simulation_results: Simulation results summary
            format: Export format ("json", "yaml")
            
        Returns:
            Path to exported file
        """
        metadata = {
            "simulation_config": simulation_config,
            "simulation_results": simulation_results,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0"
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_metadata_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif format == "yaml":
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        self.logger.info(f"Exported simulation metadata to {filepath}")
        return str(filepath)
    
    def export_research_summary(self, assessment_data: pd.DataFrame,
                               persona_data: pd.DataFrame,
                               mechanistic_data: pd.DataFrame,
                               statistical_results: Dict[str, Any],
                               format: str = "json") -> str:
        """
        Export comprehensive research summary.
        
        Args:
            assessment_data: Assessment results DataFrame
            persona_data: Persona data DataFrame
            mechanistic_data: Mechanistic analysis DataFrame
            statistical_results: Statistical analysis results
            format: Export format ("json", "yaml")
            
        Returns:
            Path to exported file
        """
        
        print("Debug: Starting export_research_summary method")
        print(f"Debug: Input types - assessment_data: {type(assessment_data)}, persona_data: {type(persona_data)}, mechanistic_data: {type(mechanistic_data)}, statistical_results: {type(statistical_results)}")
        
        # Convert DataFrames to ensure no numpy types remain
        def convert_dataframe_to_dict(df):
            """Convert DataFrame to dict with all numpy types converted."""
            if df.empty:
                return []
            return convert_numpy_types(df.to_dict('records'))
        
        # Convert DataFrames at the start
        assessment_data_dict = convert_dataframe_to_dict(assessment_data)
        persona_data_dict = convert_dataframe_to_dict(persona_data)
        mechanistic_data_dict = convert_dataframe_to_dict(mechanistic_data)
        
        # Debug: Check what's in statistical_results before conversion
        print("Debug: Checking statistical_results before conversion...")
        if isinstance(statistical_results, dict):
            for key, value in statistical_results.items():
                print(f"  Before conversion - {key}: {type(value)} = {value}")
        else:
            print(f"  statistical_results is not a dict: {type(statistical_results)}")
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            import pandas as pd
            
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif isinstance(obj, pd.DataFrame):
                # Convert DataFrame to dict with converted values
                return convert_numpy_types(obj.to_dict('records'))
            elif isinstance(obj, pd.Series):
                # Convert Series to list with converted values
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bytes_):
                return str(obj)
            elif isinstance(obj, np.str_):
                return str(obj)
            elif hasattr(obj, 'item'):  # Handle other numpy scalars
                return obj.item()
            elif hasattr(obj, 'dtype'):  # Handle pandas/numpy objects with dtype
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return str(obj)
            elif str(type(obj)).startswith("<class 'numpy."):  # Catch any remaining numpy types
                try:
                    return obj.item()
                except:
                    return str(obj)
            else:
                return obj
        
        # Convert statistical_results first to ensure no numpy types
        statistical_results = convert_numpy_types(statistical_results)
        
        # Debug: Check what's in statistical_results
        print(f"Debug: statistical_results type: {type(statistical_results)}")
        if isinstance(statistical_results, dict):
            for key, value in statistical_results.items():
                print(f"  {key}: {type(value)} = {value}")
        
        print("Debug: Creating summary dictionary...")
        
        # Create data_summary step by step
        data_summary = {}
        print("Debug: Creating data_summary...")
        
        data_summary["assessment_records"] = int(len(assessment_data_dict))
        print(f"Debug: assessment_records = {data_summary['assessment_records']} ({type(data_summary['assessment_records'])})")
        
        data_summary["persona_records"] = int(len(persona_data_dict))
        print(f"Debug: persona_records = {data_summary['persona_records']} ({type(data_summary['persona_records'])})")
        
        data_summary["mechanistic_records"] = int(len(mechanistic_data_dict))
        print(f"Debug: mechanistic_records = {data_summary['mechanistic_records']} ({type(data_summary['mechanistic_records'])})")
        
        # Get unique personas and simulation days from the original DataFrames for calculations
        unique_personas = convert_numpy_types(assessment_data['persona_id'].nunique())
        simulation_days = convert_numpy_types(assessment_data['simulation_day'].max())
        assessment_types = convert_numpy_types(assessment_data['assessment_type'].unique().tolist())
        
        data_summary["unique_personas"] = unique_personas
        print(f"Debug: unique_personas = {data_summary['unique_personas']} ({type(data_summary['unique_personas'])})")
        
        data_summary["simulation_days"] = simulation_days
        print(f"Debug: simulation_days = {data_summary['simulation_days']} ({type(data_summary['simulation_days'])})")
        
        data_summary["assessment_types"] = assessment_types
        print(f"Debug: assessment_types = {data_summary['assessment_types']} ({type(data_summary['assessment_types'])})")
        
        print("Debug: Creating data_schema...")
        
        # Create data_schema step by step
        data_schema = {}
        data_schema["assessment_data_columns"] = convert_numpy_types(assessment_data.columns.tolist())
        print(f"Debug: assessment_data_columns = {data_schema['assessment_data_columns']} ({type(data_schema['assessment_data_columns'])})")
        
        data_schema["persona_data_columns"] = convert_numpy_types(persona_data.columns.tolist())
        print(f"Debug: persona_data_columns = {data_schema['persona_data_columns']} ({type(data_schema['persona_data_columns'])})")
        
        data_schema["mechanistic_data_columns"] = convert_numpy_types(mechanistic_data.columns.tolist())
        print(f"Debug: mechanistic_data_columns = {data_schema['mechanistic_data_columns']} ({type(data_schema['mechanistic_data_columns'])})")
        
        print("Debug: Creating final summary...")
        
        summary = {
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "data_summary": data_summary,
            "statistical_results": statistical_results,
            "data_schema": data_schema,
            "assessment_data": assessment_data_dict,
            "persona_data": persona_data_dict,
            "mechanistic_data": mechanistic_data_dict
        }
        
        print("Debug: Summary created successfully")
        
        # Convert all numpy types to native Python types
        summary = convert_numpy_types(summary)
        
        # Debug: Test JSON serialization on each part individually
        print("Debug: Testing JSON serialization of each summary part...")
        
        try:
            test_summary = {"test": summary["export_timestamp"]}
            json.dumps(test_summary)
            print("Debug: export_timestamp serializes successfully")
        except Exception as e:
            print(f"Debug: export_timestamp serialization failed: {e}")
        
        try:
            test_summary = {"test": summary["export_version"]}
            json.dumps(test_summary)
            print("Debug: export_version serializes successfully")
        except Exception as e:
            print(f"Debug: export_version serialization failed: {e}")
        
        try:
            test_summary = {"test": summary["data_summary"]}
            json.dumps(test_summary)
            print("Debug: data_summary serializes successfully")
        except Exception as e:
            print(f"Debug: data_summary serialization failed: {e}")
            # Debug each part of data_summary
            for key, value in summary["data_summary"].items():
                try:
                    test_summary = {"test": value}
                    json.dumps(test_summary)
                    print(f"  Debug: data_summary[{key}] serializes successfully")
                except Exception as e:
                    print(f"  Debug: data_summary[{key}] serialization failed: {e}")
        
        try:
            test_summary = {"test": summary["statistical_results"]}
            json.dumps(test_summary)
            print("Debug: statistical_results serializes successfully")
        except Exception as e:
            print(f"Debug: statistical_results serialization failed: {e}")
        
        try:
            test_summary = {"test": summary["data_schema"]}
            json.dumps(test_summary)
            print("Debug: data_schema serializes successfully")
        except Exception as e:
            print(f"Debug: data_schema serialization failed: {e}")
            # Debug each part of data_schema
            for key, value in summary["data_schema"].items():
                try:
                    test_summary = {"test": value}
                    json.dumps(test_summary)
                    print(f"  Debug: data_schema[{key}] serializes successfully")
                except Exception as e:
                    print(f"  Debug: data_schema[{key}] serialization failed: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_summary_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        # Final safety check - convert any remaining numpy types
        try:
            # Test JSON serialization to catch any remaining issues
            json.dumps(summary)
        except TypeError as e:
            print(f"Warning: Found remaining numpy types, attempting final conversion: {e}")
            # Debug: Find the problematic value
            def find_numpy_types(obj, path=""):
                """Recursively find numpy types in the object."""
                import numpy as np
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        if str(type(value)).startswith("<class 'numpy."):
                            print(f"Found numpy type at {current_path}: {type(value)} = {value}")
                        find_numpy_types(value, current_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        current_path = f"{path}[{i}]" if path else f"[{i}]"
                        if str(type(item)).startswith("<class 'numpy."):
                            print(f"Found numpy type at {current_path}: {type(item)} = {item}")
                        find_numpy_types(item, current_path)
                elif str(type(obj)).startswith("<class 'numpy."):
                    print(f"Found numpy type at {path}: {type(obj)} = {obj}")
            
            print("Debug: Searching for numpy types in summary...")
            find_numpy_types(summary)
            summary = convert_numpy_types(summary)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
        elif format == "yaml":
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        self.logger.info(f"Exported research summary to {filepath}")
        return str(filepath)
    
    def export_for_publication(self, assessment_data: pd.DataFrame,
                              persona_data: pd.DataFrame,
                              mechanistic_data: pd.DataFrame,
                              output_dir: str = "data/results/publication") -> Dict[str, str]:
        """
        Export data in publication-ready format.
        
        Args:
            assessment_data: Assessment results DataFrame
            persona_data: Persona data DataFrame
            mechanistic_data: Mechanistic analysis DataFrame
            output_dir: Output directory for publication data
            
        Returns:
            Dictionary mapping data type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export assessment data
        assessment_file = output_path / "assessment_data.csv"
        assessment_data.to_csv(assessment_file, index=False)
        exported_files["assessment"] = str(assessment_file)
        
        # Export persona data
        persona_file = output_path / "persona_data.csv"
        persona_data.to_csv(persona_file, index=False)
        exported_files["persona"] = str(persona_file)
        
        # Export mechanistic data
        mechanistic_file = output_path / "mechanistic_data.csv"
        mechanistic_data.to_csv(mechanistic_file, index=False)
        exported_files["mechanistic"] = str(mechanistic_file)
        
        # Export data dictionary
        data_dictionary = {
            "assessment_data": {
                "description": "Psychiatric assessment results over time",
                "columns": {col: self._get_column_description(col) for col in assessment_data.columns}
            },
            "persona_data": {
                "description": "Persona baseline and state information",
                "columns": {col: self._get_column_description(col) for col in persona_data.columns}
            },
            "mechanistic_data": {
                "description": "Mechanistic analysis results",
                "columns": {col: self._get_column_description(col) for col in mechanistic_data.columns}
            }
        }
        
        dict_file = output_path / "data_dictionary.json"
        with open(dict_file, 'w') as f:
            json.dump(data_dictionary, f, indent=2)
        exported_files["dictionary"] = str(dict_file)
        
        # Export README
        readme_content = self._generate_publication_readme(assessment_data, persona_data, mechanistic_data)
        readme_file = output_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        exported_files["readme"] = str(readme_file)
        
        self.logger.info(f"Exported publication data to {output_path}")
        return exported_files
    
    def _get_column_description(self, column: str) -> str:
        """Get description for a column."""
        descriptions = {
            "assessment_id": "Unique identifier for assessment",
            "persona_id": "Unique identifier for persona",
            "assessment_type": "Type of assessment (phq9, gad7, pss10)",
            "simulation_day": "Day of simulation when assessment was conducted",
            "total_score": "Total assessment score",
            "severity_level": "Clinical severity level (minimal, mild, moderate, severe)",
            "openness": "Openness to experience trait value (0-1)",
            "conscientiousness": "Conscientiousness trait value (0-1)",
            "extraversion": "Extraversion trait value (0-1)",
            "agreeableness": "Agreeableness trait value (0-1)",
            "neuroticism": "Neuroticism trait value (0-1)",
            "drift_magnitude": "Overall personality drift magnitude",
            "attention_weight": "Attention weight from mechanistic analysis",
            "activation_value": "Activation value from mechanistic analysis"
        }
        return descriptions.get(column, "No description available")
    
    def _generate_publication_readme(self, assessment_data: pd.DataFrame,
                                   persona_data: pd.DataFrame,
                                   mechanistic_data: pd.DataFrame) -> str:
        """Generate README for publication data."""
        readme = f"""# AI Personality Drift Simulation - Publication Data

This directory contains the data files for the AI Personality Drift Simulation research.

## Data Files

### assessment_data.csv
Psychiatric assessment results over time for all personas.
- **Records**: {len(assessment_data)}
- **Unique Personas**: {assessment_data['persona_id'].nunique()}
- **Simulation Days**: {assessment_data['simulation_day'].max()}
- **Assessment Types**: {', '.join(assessment_data['assessment_type'].unique())}

### persona_data.csv
Persona baseline and state information.
- **Records**: {len(persona_data)}
- **Unique Personas**: {persona_data['persona_id'].nunique()}

### mechanistic_data.csv
Mechanistic analysis results from neural network interpretation.
- **Records**: {len(mechanistic_data)}

### data_dictionary.json
Detailed description of all variables in the dataset.

## Data Format

All data files are in CSV format with UTF-8 encoding. Missing values are represented as empty cells.

## Citation

If you use this data in your research, please cite:

```
Keeman, M. (2025). AI Personality Drift Simulation: Mechanistic Interpretability Study. 
[Dataset]. Available from: [URL]
```

## Contact

For questions about this dataset, contact: mike.keeman@gmail.com

## License

This dataset is released under the MIT License.
"""
        return readme
    
    def create_archive(self, files: List[str], archive_name: str = None) -> str:
        """
        Create a compressed archive of exported files.
        
        Args:
            files: List of file paths to archive
            archive_name: Name of archive (if None, auto-generated)
            
        Returns:
            Path to archive file
        """
        import zipfile
        
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"research_data_{timestamp}.zip"
        
        archive_path = self.output_dir / archive_name
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if Path(file_path).exists():
                    zipf.write(file_path, Path(file_path).name)
        
        self.logger.info(f"Created archive: {archive_path}")
        return str(archive_path) 