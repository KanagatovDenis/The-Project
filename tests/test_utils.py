"""
Тесты для модуля utils.py
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    export_to_html,
    export_analysis_results,
    save_visualization,
    calculate_statistics,
    format_number,
    generate_report,
    validate_config,
    load_config,
    save_config,
    create_sample_config,
    cleanup_temp_files
)
from src.data_loader import generate_sample_data
import plotly.graph_objects as go


class TestUtils:
    """Тесты для вспомогательных функций"""

    @pytest.fixture
    def sample_figure(self):
        """Фикстура с тестовым графиком"""
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
        fig.update_layout(title="Test Figure")
        return fig

    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными"""
        return generate_sample_data(num_students=10, num_weeks=4)

    def test_export_to_html(self, sample_figure, tmp_path):
        """Тест экспорта в HTML"""
        filename = tmp_path / "test_export.html"

        result = export_to_html(sample_figure, str(filename), title="Test Export")

        # Проверяем, что файл создан
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

        # Проверяем содержимое файла
        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '<!DOCTYPE html>' in content
        assert 'Test Export' in content
        assert 'plotly' in content.lower()

    def test_export_analysis_results(self, tmp_path):
        """Тест экспорта результатов анализа"""
        test_results = {
            'summary': {
                'average_grade': 7.5,
                'total_students': 100
            },
            'details': {
                'top_students': ['STD001', 'STD002']
            }
        }

        filename = tmp_path / "test_results.json"
        result = export_analysis_results(test_results, str(filename))

        # Проверяем, что файл создан
        assert Path(result).exists()

        # Проверяем содержимое файла
        with open(result, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert 'metadata' in loaded
        assert 'data' in loaded
        assert loaded['data']['summary']['average_grade'] == 7.5

    def test_save_visualization(self, sample_figure, tmp_path):
        """Тест сохранения визуализации"""
        # Тестируем разные форматы
        test_cases = [
            ('html', 'test.html'),
            ('png', 'test.png'),
        ]

        for fmt, filename in test_cases:
            filepath = tmp_path / filename

            try:
                result = save_visualization(
                    sample_figure,
                    str(filepath),
                    format=fmt,
                    width=800,
                    height=600
                )

                # Проверяем, что файл создан
                assert Path(result).exists()

                # HTML файлы должны быть больше 0 байт
                if fmt == 'html':
                    assert filepath.stat().st_size > 0

            except Exception as e:
                # PNG может требовать дополнительных зависимостей
                if fmt == 'png' and 'kaleido' in str(e).lower():
                    print(f"⚠️  PNG экспорт требует kaleido: {e}")
                    continue
                else:
                    raise e

    def test_calculate_statistics(self):
        """Тест расчета статистики"""
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        stats = calculate_statistics(test_data)

        # Проверяем структуру
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

        # Проверяем значения
        assert stats['mean'] == 5.5
        assert stats['median'] == 5.5
        assert stats['min'] == 1.0
        assert stats['max'] == 10.0
        assert stats['count'] == 10

        # Тест с пустыми данными
        empty_stats = calculate_statistics([])
        assert empty_stats == {}

    def test_format_number(self):
        """Тест форматирования чисел"""
        test_cases = [
            (1234567, 2, '1.23M'),
            (1234, 2, '1.23K'),
            (1.23456e-5, 3, '1.235e-05'),
            (42.12345, 2, '42.12'),
            (0, 2, '0.00'),
            (None, 2, 'N/A')
        ]

        for value, decimals, expected in test_cases:
            result = format_number(value, decimals)
            assert result == expected

    def test_generate_report(self, sample_data):
        """Тест генерации отчета"""
        report = generate_report(sample_data, report_type='weekly')

        # Проверяем структуру
        assert 'metadata' in report
        assert 'summary' in report
        assert 'details' in report
        assert 'recommendations' in report

        # Проверяем метаданные
        assert report['metadata']['report_type'] == 'weekly'
        assert 'generated_at' in report['metadata']

        # Проверяем сводку
        assert 'total_students' in report['summary']
        assert 'average_grade' in report['summary']

        # Проверяем детали
        assert 'top_subjects' in report['details']
        assert 'top_students' in report['details']

        # Проверяем значения
        assert report['summary']['total_students'] == sample_data['student_id'].nunique()
        assert 1 <= report['summary']['average_grade'] <= 10

    def test_config_management(self, tmp_path):
        """Тест управления конфигурацией"""
        # Создаем тестовую конфигурацию
        config = create_sample_config()

        # Проверяем структуру
        assert 'data_source' in config
        assert 'risk_threshold' in config
        assert 'visualization' in config

        # Проверяем валидацию
        assert validate_config(config) == True

        # Тест с некорректной конфигурацией
        invalid_config = {'data_source': 'test'}
        assert validate_config(invalid_config) == False

        # Тест сохранения и загрузки
        config_file = tmp_path / "test_config.json"

        # Сохраняем
        save_success = save_config(config, str(config_file))
        assert save_success == True
        assert config_file.exists()

        # Загружаем
        loaded_config = load_config(str(config_file))
        assert loaded_config == config

    def test_cleanup_temp_files(self, tmp_path):
        """Тест очистки временных файлов"""
        temp_dir = tmp_path / "temp_test"
        temp_dir.mkdir()

        # Создаем тестовые файлы
        files_created = 0
        for i in range(5):
            file_path = temp_dir / f"test_{i}.txt"
            file_path.write_text(f"Test content {i}")
            files_created += 1

        # Очищаем файлы
        deleted_count = cleanup_temp_files(str(temp_dir), max_age_days=0)

        # Проверяем, что все файлы удалены
        assert deleted_count == files_created
        assert len(list(temp_dir.glob('*'))) == 0

    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Тест с некорректными путями
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = Path(tmpdir) / "nonexistent" / "file.json"

            # Должен вернуть пустой словарь при ошибке загрузки
            config = load_config(str(non_existent))
            assert config == {}

            # Должен вернуть False при ошибке сохранения
            save_success = save_config({}, str(non_existent.parent))
            assert save_success == False


class TestIntegration:
    """Интеграционные тесты"""

    def test_end_to_end_workflow(self, tmp_path):
        """Тест сквозного рабочего процесса"""
        # 1. Генерируем данные
        from src.data_loader import generate_sample_data
        data = generate_sample_data(num_students=20, num_weeks=8)

        # 2. Анализируем данные
        from src.analyzer import analyze_performance
        analysis = analyze_performance(data)

        # 3. Создаем визуализацию
        from src.visualizer import create_grade_distribution
        fig = create_grade_distribution(data)

        # 4. Экспортируем результаты
        from src.utils import export_analysis_results, save_visualization

        analysis_file = tmp_path / "analysis.json"
        export_analysis_results(analysis, str(analysis_file))

        visualization_file = tmp_path / "visualization.html"
        save_visualization(fig, str(visualization_file), format='html')

        # 5. Проверяем результаты
        assert analysis_file.exists()
        assert visualization_file.exists()

        # Проверяем содержимое файла анализа
        with open(analysis_file, 'r', encoding='utf-8') as f:
            loaded_analysis = json.load(f)
            assert 'data' in loaded_analysis
            assert 'overall' in loaded_analysis['data']

    def test_config_integration(self):
        """Тест интеграции с конфигурацией"""
        config = create_sample_config()

        # Проверяем, что конфигурация содержит все необходимые поля
        required_sections = ['data_source', 'analysis_period', 'risk_threshold']
        for section in required_sections:
            assert section in config

        # Проверяем типы данных
        assert isinstance(config['risk_threshold'], float)
        assert isinstance(config['min_records_per_student'], int)
        assert isinstance(config['visualization'], dict)