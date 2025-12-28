"""
Тесты для модуля data_loader.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_student_data, clean_data, generate_sample_data, validate_data


class TestDataLoader:
    """Тесты для загрузки данных"""

    def test_generate_sample_data(self):
        """Тест генерации тестовых данных"""
        df = generate_sample_data(num_students=10, num_weeks=4)

        # Проверяем структуру данных
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'student_id' in df.columns
        assert 'grade' in df.columns
        assert 'subject' in df.columns

        # Проверяем диапазон оценок
        assert df['grade'].min() >= 1
        assert df['grade'].max() <= 10

        # Проверяем уникальность студентов
        assert df['student_id'].nunique() == 10

    def test_clean_data(self):
        """Тест очистки данных"""
        # Создаем тестовые данные с проблемами
        test_data = pd.DataFrame({
            'student_id': ['STD001', 'STD001', 'STD002', 'STD003', 'STD003'],
            'grade': [8.5, 8.5, 12.0, -1.0, 7.5],  # Некорректные оценки
            'subject': ['Math', 'Math', 'Physics', 'Math', 'Physics'],
            'attendance': [1.0, 1.0, 0.5, None, 0.8]  # Пропущенное значение
        })

        cleaned = clean_data(test_data)

        # Проверяем, что некорректные оценки удалены
        assert cleaned['grade'].min() >= 1
        assert cleaned['grade'].max() <= 10

        # Проверяем обработку пропущенных значений
        assert cleaned['attendance'].isna().sum() == 0

        # Проверяем удаление дубликатов
        assert len(cleaned) <= len(test_data)

    def test_validate_data(self):
        """Тест валидации данных"""
        df = generate_sample_data(num_students=5, num_weeks=2)
        report = validate_data(df)

        # Проверяем структуру отчета
        assert 'total_records' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'data_types' in report

        # Проверяем значения
        assert report['total_records'] == len(df)
        assert report['total_columns'] == len(df.columns)

    def test_load_nonexistent_file(self):
        """Тест загрузки несуществующего файла"""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = Path(tmpdir) / 'nonexistent.csv'

            with pytest.raises(FileNotFoundError):
                load_student_data(str(non_existent))

    def test_data_integrity(self):
        """Тест целостности данных"""
        df = generate_sample_data(num_students=20, num_weeks=8)

        # Проверяем обязательные колонки
        required_columns = ['student_id', 'grade', 'subject']
        for col in required_columns:
            assert col in df.columns

        # Проверяем отсутствие NaN в обязательных колонках
        for col in required_columns:
            assert df[col].notna().all()

        # Проверяем корректность типов данных
        assert pd.api.types.is_string_dtype(df['student_id'])
        assert pd.api.types.is_numeric_dtype(df['grade'])

        # Проверяем логическую целостность
        assert (df['grade'] >= 1).all()
        assert (df['grade'] <= 10).all()

        if 'attendance' in df.columns:
            assert (df['attendance'] >= 0).all()
            assert (df['attendance'] <= 1).all()


class TestPerformance:
    """Тесты производительности"""

    def test_large_dataset(self):
        """Тест обработки большого набора данных"""
        # Генерируем большой набор данных
        df = generate_sample_data(num_students=1000, num_weeks=16)

        # Проверяем, что данные сгенерированы
        assert len(df) > 10000

        # Тест очистки с таймингом
        import time
        start_time = time.time()

        cleaned = clean_data(df)

        end_time = time.time()
        processing_time = end_time - start_time

        # Проверяем, что обработка заняла разумное время
        assert processing_time < 10.0  # 10 секунд

        # Проверяем, что данные остались целыми
        assert len(cleaned) > 0
        assert 'student_id' in cleaned.columns

    def test_memory_usage(self):
        """Тест использования памяти"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # в MB

        # Генерируем данные
        df = generate_sample_data(num_students=500, num_weeks=12)

        memory_during = process.memory_info().rss / 1024 / 1024

        # Очищаем данные
        cleaned = clean_data(df)

        # Освобождаем память
        del df
        del cleaned

        memory_after = process.memory_info().rss / 1024 / 1024

        # Проверяем, что память не утекла
        assert memory_after - memory_before < 100  # Не более 100MB утечки