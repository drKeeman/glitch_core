"""
Memory visualization tools for analysis and debugging.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from glitch_core.config.logging import get_logger
from .manager import MemoryRecord
from .compressor import CompressedMemory
from .temporal_decay import TemporalDecay
from .relevance_scorer import RelevanceScore


@dataclass
class MemoryVisualizationData:
    """Data structure for memory visualization."""
    memories: List[MemoryRecord]
    compressed_memories: List[CompressedMemory]
    relevance_scores: List[RelevanceScore]
    decay_strengths: Dict[str, float]
    time_range: Tuple[datetime, datetime]
    memory_types: List[str]
    emotional_dimensions: List[str]


class MemoryVisualizer:
    """
    Creates visualizations for memory analysis and debugging.
    """
    
    def __init__(self):
        self.logger = get_logger("memory_visualizer")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_memory_timeline(
        self, 
        memories: List[MemoryRecord],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        memory_types: Optional[List[str]] = None
    ) -> Figure:
        """
        Create a timeline visualization of memories.
        
        Args:
            memories: List of memories to visualize
            time_range: Optional time range to focus on
            memory_types: Optional list of memory types to include
            
        Returns:
            Matplotlib figure with timeline
        """
        if not memories:
            return self._create_empty_figure("No memories to visualize")
        
        # Filter memories
        filtered_memories = self._filter_memories(memories, time_range, memory_types)
        
        if not filtered_memories:
            return self._create_empty_figure("No memories in specified range")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        timestamps = [m.timestamp for m in filtered_memories]
        emotional_weights = [m.emotional_weight for m in filtered_memories]
        memory_types = [m.memory_type for m in filtered_memories]
        
        # Create scatter plot
        scatter = ax.scatter(
            timestamps, 
            emotional_weights,
            c=[self._get_memory_type_color(mt) for mt in memory_types],
            s=[self._get_memory_size(m) for m in filtered_memories],
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Emotional Weight')
        ax.set_title('Memory Timeline')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        unique_types = list(set(memory_types))
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self._get_memory_type_color(mt),
                      markersize=10, label=mt)
            for mt in unique_types
        ]
        ax.legend(handles=legend_elements, title='Memory Types')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_emotional_distribution(
        self, 
        memories: List[MemoryRecord]
    ) -> Figure:
        """
        Create a distribution visualization of emotional weights.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            Matplotlib figure with emotional distribution
        """
        if not memories:
            return self._create_empty_figure("No memories to analyze")
        
        # Extract emotional weights
        emotional_weights = [m.emotional_weight for m in memories]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(emotional_weights, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Emotional Weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Emotional Weights')
        ax1.grid(True, alpha=0.3)
        
        # Box plot by memory type
        memory_types = list(set(m.memory_type for m in memories))
        type_weights = []
        type_labels = []
        
        for mt in memory_types:
            mt_memories = [m for m in memories if m.memory_type == mt]
            if mt_memories:
                weights = [m.emotional_weight for m in mt_memories]
                type_weights.append(weights)
                type_labels.append(mt)
        
        if type_weights:
            ax2.boxplot(type_weights, labels=type_labels)
            ax2.set_xlabel('Memory Type')
            ax2.set_ylabel('Emotional Weight')
            ax2.set_title('Emotional Weights by Memory Type')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_decay_analysis(
        self, 
        memories: List[MemoryRecord],
        current_time: datetime,
        temporal_decay: TemporalDecay
    ) -> Figure:
        """
        Create a visualization of memory decay patterns.
        
        Args:
            memories: List of memories to analyze
            current_time: Current time for decay calculation
            temporal_decay: Temporal decay calculator
            
        Returns:
            Matplotlib figure with decay analysis
        """
        if not memories:
            return self._create_empty_figure("No memories to analyze")
        
        # Calculate decay strengths
        decay_strengths = temporal_decay.calculate_decay_batch(memories, current_time)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        ages_hours = [(current_time - m.timestamp).total_seconds() / 3600 for m in memories]
        strengths = [decay_strengths[m.id] for m in memories]
        memory_types = [m.memory_type for m in memories]
        
        # 1. Age vs Strength scatter plot
        scatter = ax1.scatter(
            ages_hours, 
            strengths,
            c=[self._get_memory_type_color(mt) for mt in memory_types],
            alpha=0.7,
            s=50
        )
        ax1.set_xlabel('Age (hours)')
        ax1.set_ylabel('Decay Strength')
        ax1.set_title('Memory Decay Over Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Decay by memory type
        unique_types = list(set(memory_types))
        type_strengths = {}
        for mt in unique_types:
            type_strengths[mt] = [strengths[i] for i, m in enumerate(memories) if m.memory_type == mt]
        
        if type_strengths:
            ax2.boxplot(type_strengths.values(), labels=type_strengths.keys())
            ax2.set_xlabel('Memory Type')
            ax2.set_ylabel('Decay Strength')
            ax2.set_title('Decay Strength by Memory Type')
            ax2.grid(True, alpha=0.3)
        
        # 3. Strength distribution
        ax3.hist(strengths, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Decay Strength')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Decay Strengths')
        ax3.grid(True, alpha=0.3)
        
        # 4. Age distribution
        ax4.hist(ages_hours, bins=20, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Age (hours)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Memory Ages')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_relevance_analysis(
        self, 
        relevance_scores: List[RelevanceScore]
    ) -> Figure:
        """
        Create a visualization of relevance scoring results.
        
        Args:
            relevance_scores: List of relevance scores
            
        Returns:
            Matplotlib figure with relevance analysis
        """
        if not relevance_scores:
            return self._create_empty_figure("No relevance scores to analyze")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        total_scores = [score.total_score for score in relevance_scores]
        content_scores = [score.content_relevance for score in relevance_scores]
        emotional_scores = [score.emotional_relevance for score in relevance_scores]
        temporal_scores = [score.temporal_relevance for score in relevance_scores]
        contextual_scores = [score.contextual_relevance for score in relevance_scores]
        persona_scores = [score.persona_relevance for score in relevance_scores]
        
        # 1. Total score distribution
        ax1.hist(total_scores, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Total Relevance Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Total Relevance Scores')
        ax1.grid(True, alpha=0.3)
        
        # 2. Factor comparison
        factors = ['Content', 'Emotional', 'Temporal', 'Contextual', 'Persona']
        factor_scores = [content_scores, emotional_scores, temporal_scores, contextual_scores, persona_scores]
        factor_means = [np.mean(scores) for scores in factor_scores]
        
        bars = ax2.bar(factors, factor_means, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Relevance by Factor')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, factor_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 3. Correlation heatmap
        correlation_data = np.array([
            content_scores, emotional_scores, temporal_scores, 
            contextual_scores, persona_scores, total_scores
        ])
        correlation_matrix = np.corrcoef(correlation_data)
        
        im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_xticks(range(len(factors) + 1))
        ax3.set_yticks(range(len(factors) + 1))
        ax3.set_xticklabels(factors + ['Total'], rotation=45)
        ax3.set_yticklabels(factors + ['Total'])
        ax3.set_title('Factor Correlation Matrix')
        
        # Add correlation values
        for i in range(len(factors) + 1):
            for j in range(len(factors) + 1):
                text = ax3.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black")
        
        # 4. Score scatter plot (Content vs Emotional)
        ax4.scatter(content_scores, emotional_scores, alpha=0.7, s=50)
        ax4.set_xlabel('Content Relevance')
        ax4.set_ylabel('Emotional Relevance')
        ax4.set_title('Content vs Emotional Relevance')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(content_scores) > 1:
            z = np.polyfit(content_scores, emotional_scores, 1)
            p = np.poly1d(z)
            ax4.plot(content_scores, p(content_scores), "r--", alpha=0.8)
        
        plt.tight_layout()
        return fig
    
    def create_compression_analysis(
        self, 
        original_memories: List[MemoryRecord],
        compressed_memories: List[CompressedMemory]
    ) -> Figure:
        """
        Create a visualization of memory compression results.
        
        Args:
            original_memories: Original memories before compression
            compressed_memories: Compressed memories after compression
            
        Returns:
            Matplotlib figure with compression analysis
        """
        if not original_memories or not compressed_memories:
            return self._create_empty_figure("No compression data to analyze")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        original_count = len(original_memories)
        compressed_count = len(compressed_memories)
        compression_ratios = [cm.compression_ratio for cm in compressed_memories]
        memory_counts = [cm.memory_count for cm in compressed_memories]
        
        # 1. Compression overview
        labels = ['Original', 'Compressed']
        sizes = [original_count, compressed_count]
        colors = ['lightblue', 'lightcoral']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Memory Compression Overview')
        
        # 2. Compression ratio distribution
        ax2.hist(compression_ratios, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Compression Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Compression Ratios')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory count per compressed memory
        ax3.hist(memory_counts, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Memories per Compressed Memory')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Memories per Compressed Memory')
        ax3.grid(True, alpha=0.3)
        
        # 4. Compression ratio vs memory count
        ax4.scatter(memory_counts, compression_ratios, alpha=0.7, s=50)
        ax4.set_xlabel('Memory Count')
        ax4.set_ylabel('Compression Ratio')
        ax4.set_title('Compression Ratio vs Memory Count')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _filter_memories(
        self, 
        memories: List[MemoryRecord],
        time_range: Optional[Tuple[datetime, datetime]],
        memory_types: Optional[List[str]]
    ) -> List[MemoryRecord]:
        """Filter memories based on time range and types."""
        filtered = memories
        
        if time_range:
            start_time, end_time = time_range
            filtered = [m for m in filtered if start_time <= m.timestamp <= end_time]
        
        if memory_types:
            filtered = [m for m in filtered if m.memory_type in memory_types]
        
        return filtered
    
    def _get_memory_type_color(self, memory_type: str) -> str:
        """Get color for memory type."""
        colors = {
            'event': 'blue',
            'reflection': 'green',
            'intervention': 'red',
            'trauma': 'darkred',
            'success': 'gold',
            'default': 'gray'
        }
        return colors.get(memory_type, colors['default'])
    
    def _get_memory_size(self, memory: MemoryRecord) -> float:
        """Get size for memory point based on emotional weight."""
        return 50 + memory.emotional_weight * 100
    
    def _create_empty_figure(self, message: str) -> Figure:
        """Create an empty figure with a message."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    def save_visualization(self, fig: Figure, filename: str, dpi: int = 300):
        """Save visualization to file."""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            self.logger.info("visualization_saved", filename=filename)
        except Exception as e:
            self.logger.error("visualization_save_failed", filename=filename, error=str(e))
    
    def create_memory_summary_report(
        self, 
        visualization_data: MemoryVisualizationData
    ) -> Dict[str, Any]:
        """
        Create a comprehensive memory summary report.
        
        Args:
            visualization_data: Data for visualization
            
        Returns:
            Dictionary with summary statistics
        """
        memories = visualization_data.memories
        compressed_memories = visualization_data.compressed_memories
        relevance_scores = visualization_data.relevance_scores
        
        if not memories:
            return {"error": "No memories to analyze"}
        
        # Basic statistics
        total_memories = len(memories)
        memory_types = list(set(m.memory_type for m in memories))
        time_span = (min(m.timestamp for m in memories), max(m.timestamp for m in memories))
        
        # Emotional statistics
        emotional_weights = [m.emotional_weight for m in memories]
        avg_emotional_weight = np.mean(emotional_weights)
        emotional_std = np.std(emotional_weights)
        
        # Type distribution
        type_counts = {}
        for mt in memory_types:
            type_counts[mt] = len([m for m in memories if m.memory_type == mt])
        
        # Compression statistics
        compression_stats = {}
        if compressed_memories:
            compression_ratios = [cm.compression_ratio for cm in compressed_memories]
            compression_stats = {
                "compressed_count": len(compressed_memories),
                "average_compression_ratio": np.mean(compression_ratios),
                "compression_efficiency": len(compressed_memories) / total_memories
            }
        
        # Relevance statistics
        relevance_stats = {}
        if relevance_scores:
            total_scores = [rs.total_score for rs in relevance_scores]
            relevance_stats = {
                "average_relevance": np.mean(total_scores),
                "high_relevance_count": len([s for s in total_scores if s > 0.7]),
                "low_relevance_count": len([s for s in total_scores if s < 0.3])
            }
        
        return {
            "total_memories": total_memories,
            "memory_types": memory_types,
            "type_distribution": type_counts,
            "time_span": {
                "start": time_span[0].isoformat(),
                "end": time_span[1].isoformat(),
                "duration_hours": (time_span[1] - time_span[0]).total_seconds() / 3600
            },
            "emotional_statistics": {
                "average_weight": avg_emotional_weight,
                "std_weight": emotional_std,
                "min_weight": min(emotional_weights),
                "max_weight": max(emotional_weights)
            },
            "compression_statistics": compression_stats,
            "relevance_statistics": relevance_stats
        } 