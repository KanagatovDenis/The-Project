"""
Тесты для модуля visualizer.py
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import (
    create_grade_distribution,
    create_performance_trend,
    create_group_comparison,
    create_correlation_matrix,
    create_risk_students_plot,
    create_subject_analysis,
    create_student_portfolio,
    save_visualization
)
from src.data_loader import generate_sample_data


class TestVisualizer:
    """Тесты для визуализаций"""

    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными"""
        return generate_sample_data(num_students=30, num_weeks=8)

    def test_create_grade_distribution(self, sample_data):
        """Тест создания распределения оценок"""
        fig = create_grade_distribution(sample_data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Проверяем наличие осей
        assert fig.layout.xaxis.title.text is not None
        assert fig.layout.yaxis.title.text is not None

        # Тест с фильтром по предмету
        subject = sample_data['subject'].iloc[0]
        fig_filtered = create_grade_distribution(sample_data, subject=subject)
        assert isinstance(fig_filtered, go.Figure)

    def test_create_performance_trend(self, sample_data):
        """Тест создания графика тренда"""
        # Добавляем недели если их нет
        if 'week' not in sample_data.columns:
            sample_data['week'] = sample_data['date'].dt.isocalendar().week

        # Выбираем несколько студентов
        student_ids = sample_data['student_id'].unique()[:3]

        fig = create_performance_trend(sample_data, student_ids=student_ids)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Проверяем, что линии созданы для каждого студента
        assert len(fig.data) >= len(student_ids) * 2  # Линия + маркеры

    def test_create_group_comparison(self, sample_data):
        """Тест сравнения групп"""
        if 'group' not in sample_data.columns:
            sample_data['group'] = ['ГРП-1', 'ГРП-2', 'ГРП-3'] * (len(sample_data) // 3 + 1)
            sample_data['group'] = sample_data['group'][:len(sample_data)]

        fig = create_group_comparison(sample_data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_correlation_matrix(self, sample_data):
        """Тест создания матрицы корреляции"""
        # Ограничиваем количество предметов
        subjects = sample_data['subject'].unique()[:4]
        filtered_data = sample_data[sample_data['subject'].isin(subjects)]

        fig = create_correlation_matrix(filtered_data, subjects=subjects)

        assert isinstance(fig, go.Figure)

        # Проверяем, что это heatmap
        if len(fig.data) > 0:
            assert fig.data[0].type == 'heatmap'

    def test_create_risk_students_plot(self, sample_data):
        """Тест создания графика студентов группы риска"""
        fig = create_risk_students_plot(sample_data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Проверяем наличие scatter plot
        if len(fig.data) > 0:
            assert fig.data[0].type == 'scatter'

    def test_create_subject_analysis(self, sample_data):
        """Тест комплексного анализа предметов"""
        fig = create_subject_analysis(sample_data)

        assert isinstance(fig, go.Figure)

        # Проверяем, что это subplot
        assert hasattr(fig, 'get_subplot_rows')
        assert hasattr(fig, 'get_subplot_cols')

    def test_create_student_portfolio(self, sample_data):
        """Тест создания портфолио студента"""
        student_id = sample_data['student_id'].iloc[0]

        fig = create_student_portfolio(student_id, sample_data)

        assert isinstance(fig, go.Figure)

        # Проверяем, что это subplot
        assert hasattr(fig, 'get_subplot_rows')
        assert hasattr(fig, 'get_subplot_cols')

    def test_save_visualization(self, sample_data, tmp_path):
        """Тест сохранения визуализации"""
        fig = create_grade_distribution(sample_data)

        # Тестируем сохранение в разных форматах
        test_formats = ['html', 'png']

        for fmt in test_formats:
            filename = tmp_path / f'test_visualization.{fmt}'

            try:
                save_visualization(fig, str(filename), format=fmt)

                # Проверяем, что файл создан
                assert filename.exists()
                assert filename.stat().st_size > 0

                print(f"✅ Файл {fmt} успешно создан: {filename.stat().st_size} байт")

            except Exception as e:
                # PNG может требовать дополнительных зависимостей
                if fmt == 'png' and 'kaleido' in str(e).lower():
                    print(f"⚠️  PNG экспорт требует установки kaleido: {e}")
                    continue
                else:
                    raise e

    def test_empty_data(self):
        """Тест обработки пустых данных"""
        empty_df = pd.DataFrame()

        # Все функции должны корректно обрабатывать пустые данные
        with pytest.raises(ValueError):
            create_grade_distribution(empty_df)

        with pytest.raises(ValueError):
            create_performance_trend(empty_df)

    def test_invalid_parameters(self, sample_data):
        """Тест с некорректными параметрами"""
        # Некорректный студент
        with pytest.raises(ValueError):
            create_student_portfolio('NON_EXISTENT', sample_data)

        # Некорректный предмет
        fig = create_grade_distribution(sample_data, subject='NON_EXISTENT')
        # Должен вернуть пустой график или ошибку

        # Некорректная группа
        if 'group' in sample_data.columns:
            fig = create_group_comparison(sample_data[sample_data['group'] == 'NON_EXISTENT'])
            # Должен корректно обработать отсутствие данных


class TestVisualizationQuality:
    """Тесты качества визуализаций"""

    @pytest.fixture
    def complex_data(self):
        """Фикстура с комплексными данными для тестов"""
        return generate_sample_data(num_students=50, num_weeks=12)

    def test_color_schemes(self, complex_data):
        """Тест различных цветовых схем"""
        fig = create_grade_distribution(complex_data)

        # Проверяем, что график имеет цвета
        if len(fig.data) > 0:
            assert hasattr(fig.data[0], 'marker')

        # Проверяем наличие заголовка
        assert fig.layout.title.text is not None

    def test_layout_configuration(self, complex_data):
        """Тест конфигурации макета"""
        fig = create_subject_analysis(complex_data)

        # Проверяем настройки макета
        assert fig.layout.showlegend in [True, False]
        assert fig.layout.plot_bgcolor is not None
        assert fig.layout.font is not None

    def test_interactive_elements(self, complex_data):
        """Тест интерактивных элементов"""
        fig = create_risk_students_plot(complex_data)

        # Проверяем наличие hover информации
        if len(fig.data) > 0:
            assert fig.data[0].hoverinfo is not None

        # Проверяем наличие аннотаций
        assert len(fig.layout.annotations) > 0

    def test_data_integrity_in_visualizations(self, complex_data):
        """Тест целостности данных в визуализациях"""
        # Создаем разные визуализации
        visualizations = [
            create_grade_distribution(complex_data),
            create_performance_trend(complex_data),
            create_group_comparison(complex_data),
            create_risk_students_plot(complex_data)
        ]

        for fig in visualizations:
            # Проверяем, что визуализация создана
            assert isinstance(fig, go.Figure)

            # Проверяем, что есть данные
            assert len(fig.data) > 0 or len(fig.layout.annotations) > 0

            # Проверяем отсутствие NaN в данных визуализации
            for trace in fig.data:
                if hasattr(trace, 'x'):
                    assert not any(pd.isna(trace.x))
                if hasattr(trace, 'y'):
                    assert not any(pd.isna(trace.y))