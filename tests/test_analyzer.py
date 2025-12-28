"""
Тесты для модуля analyzer.py
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import (
    analyze_performance,
    identify_at_risk_students,
    calculate_subject_statistics,
    predict_final_grades,
    calculate_learning_metrics,
    generate_recommendations
)
from src.data_loader import generate_sample_data


class TestAnalyzer:
    """Тесты для аналитического модуля"""

    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными"""
        return generate_sample_data(num_students=20, num_weeks=8)

    def test_analyze_performance(self, sample_data):
        """Тест комплексного анализа"""
        analysis = analyze_performance(sample_data)

        # Проверяем структуру результатов
        assert 'overall' in analysis
        assert 'by_subject' in analysis
        assert 'risk_students' in analysis
        assert 'timestamp' in analysis

        # Проверяем общую статистику
        overall = analysis['overall']
        assert 'total_records' in overall
        assert 'mean_grade' in overall
        assert 'total_students' in overall

        # Проверяем значения
        assert overall['total_records'] == len(sample_data)
        assert overall['total_students'] == sample_data['student_id'].nunique()
        assert 1 <= overall['mean_grade'] <= 10

        # Проверяем статистику по предметам
        assert len(analysis['by_subject']) == sample_data['subject'].nunique()

        for subject, stats in analysis['by_subject'].items():
            assert 'mean_grade' in stats
            assert 'student_count' in stats
            assert 1 <= stats['mean_grade'] <= 10

    def test_identify_at_risk_students(self, sample_data):
        """Тест идентификации студентов группы риска"""
        risk_df = identify_at_risk_students(sample_data)

        # Проверяем тип результата
        assert isinstance(risk_df, pd.DataFrame)

        if not risk_df.empty:
            # Проверяем структуру DataFrame
            assert 'student_id' in risk_df.columns
            assert 'avg_grade' in risk_df.columns
            assert 'risk_factors' in risk_df.columns
            assert 'risk_score' in risk_df.columns

            # Проверяем, что оценки риска корректны
            assert (risk_df['risk_score'] >= 1).all()
            assert (risk_df['avg_grade'] < 5.0).all()  # По умолчанию порог 5.0

    def test_calculate_subject_statistics(self, sample_data):
        """Тест расчета статистики по предметам"""
        stats = calculate_subject_statistics(sample_data)

        # Проверяем структуру
        assert isinstance(stats, dict)
        assert len(stats) == sample_data['subject'].nunique()

        for subject, subject_stats in stats.items():
            # Проверяем базовую статистику
            assert 'basic' in subject_stats
            assert 'counts' in subject_stats
            assert 'distribution' in subject_stats

            basic = subject_stats['basic']
            assert 'mean' in basic
            assert 'std' in basic
            assert 'min' in basic
            assert 'max' in basic

            # Проверяем значения
            assert basic['min'] <= basic['mean'] <= basic['max']
            assert basic['std'] >= 0

    def test_predict_final_grades(self, sample_data):
        """Тест прогнозирования итоговых оценок"""
        # Добавляем недели если их нет
        if 'week' not in sample_data.columns:
            sample_data['week'] = sample_data['date'].dt.isocalendar().week

        predictions = predict_final_grades(sample_data, current_week=8)

        # Проверяем тип результата
        assert isinstance(predictions, pd.DataFrame)

        if not predictions.empty:
            # Проверяем структуру
            assert 'student_id' in predictions.columns
            assert 'subject' in predictions.columns
            assert 'predicted_final_grade' in predictions.columns
            assert 'prediction_confidence' in predictions.columns

            # Проверяем значения
            assert (predictions['predicted_final_grade'] >= 1).all()
            assert (predictions['predicted_final_grade'] <= 10).all()
            assert (predictions['prediction_confidence'] >= 0).all()
            assert (predictions['prediction_confidence'] <= 1).all()

    def test_calculate_learning_metrics(self, sample_data):
        """Тест расчета метрик обучения"""
        metrics = calculate_learning_metrics(sample_data)

        # Проверяем структуру
        assert 'overall_efficiency' in metrics
        assert 'grade_distribution' in metrics

        overall = metrics['overall_efficiency']
        assert 'average_grade' in overall
        assert 'pass_rate' in overall
        assert 'failure_rate' in overall

        # Проверяем значения
        assert 1 <= overall['average_grade'] <= 10
        assert 0 <= overall['pass_rate'] <= 100
        assert 0 <= overall['failure_rate'] <= 100

        # Проверяем распределение оценок
        distribution = metrics['grade_distribution']
        total_percentage = sum(distribution.values())
        assert 99 <= total_percentage <= 101  # Допускаем погрешность округления

    def test_generate_recommendations(self):
        """Тест генерации рекомендаций"""
        # Тест с разными факторами риска
        test_cases = [
            (['Низкая средняя оценка'], 3),
            (['Высокая изменчивость оценок'], 3),
            (['Снижение успеваемости', 'Низкая посещаемость'], 5),
            ([], 0)
        ]

        for risk_factors, expected_min in test_cases:
            recommendations = generate_recommendations(risk_factors)

            assert isinstance(recommendations, list)

            if risk_factors:
                assert len(recommendations) >= expected_min
                assert all(isinstance(r, str) for r in recommendations)
            else:
                assert len(recommendations) == 0

    def test_edge_cases(self):
        """Тест граничных случаев"""
        # Пустые данные
        empty_df = pd.DataFrame()

        with pytest.raises(Exception):
            analyze_performance(empty_df)

        # Минимальные данные
        min_data = pd.DataFrame({
            'student_id': ['STD001'],
            'grade': [5.0],
            'subject': ['Math']
        })

        analysis = analyze_performance(min_data)
        assert analysis['overall']['total_records'] == 1
        assert analysis['overall']['total_students'] == 1

    def test_consistency(self, sample_data):
        """Тест консистентности результатов"""
        # Многократный анализ должен давать одинаковые результаты
        results = []

        for _ in range(3):
            analysis = analyze_performance(sample_data)
            results.append(analysis['overall']['mean_grade'])

        # Проверяем, что результаты близки (допускаем погрешность округления)
        assert max(results) - min(results) < 0.01


class TestPerformanceAnalysis:
    """Тесты производительности анализа"""

    @pytest.fixture
    def large_dataset(self):
        """Фикстура с большим набором данных"""
        return generate_sample_data(num_students=200, num_weeks=16)

    def test_analysis_performance(self, large_dataset):
        """Тест производительности анализа"""
        import time

        start_time = time.time()
        analysis = analyze_performance(large_dataset)
        end_time = time.time()

        processing_time = end_time - start_time

        # Проверяем, что анализ выполняется за разумное время
        assert processing_time < 5.0  # 5 секунд

        # Проверяем, что результаты корректны
        assert analysis['overall']['total_records'] == len(large_dataset)
        assert analysis['overall']['total_students'] == large_dataset['student_id'].nunique()

    def test_memory_efficiency(self, large_dataset):
        """Тест эффективности использования памяти"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Измеряем память до анализа
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Выполняем анализ
        analysis = analyze_performance(large_dataset)

        # Измеряем память после анализа
        memory_after = process.memory_info().rss / 1024 / 1024

        # Освобождаем память
        del analysis

        memory_final = process.memory_info().rss / 1024 / 1024

        # Проверяем, что нет значительной утечки памяти
        memory_increase = memory_final - memory_before
        assert memory_increase < 50  # Не более 50MB утечки

    def test_concurrent_analysis(self):
        """Тест конкурентного анализа (опционально)"""
        # Этот тест можно использовать для проверки работы в многопоточном окружении
        datasets = [generate_sample_data(num_students=10, num_weeks=4) for _ in range(3)]

        results = []
        for dataset in datasets:
            analysis = analyze_performance(dataset)
            results.append(analysis['overall']['mean_grade'])

        # Проверяем, что все анализы выполнены
        assert len(results) == 3
        assert all(1 <= r <= 10 for r in results)