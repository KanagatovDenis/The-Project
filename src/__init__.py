"""
EduViz Dashboard - система визуализации образовательных данных
"""

__version__ = '1.0.0'
__author__ = 'УрФУ студент'

from .data_loader import load_student_data, clean_data, merge_datasets
from .visualizer import (
    create_grade_distribution,
    create_performance_trend,
    create_group_comparison,
    create_correlation_matrix,
    create_risk_students_plot
)
from .analyzer import (
    analyze_performance,
    identify_at_risk_students,
    calculate_subject_statistics,
    predict_final_grades
)
from .dashboard import create_dashboard
from .utils import export_to_html, save_visualization

__all__ = [
    'load_student_data',
    'create_grade_distribution',
    'analyze_performance',
    'create_dashboard'
]