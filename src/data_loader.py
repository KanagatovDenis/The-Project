"""
Модуль для загрузки и обработки образовательных данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_student_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Загружает данные об успеваемости студентов из CSV файла.

    Parameters:
    -----------
    filepath : str
        Путь к CSV файлу с данными
    **kwargs : dict
        Дополнительные параметры для pd.read_csv

    Returns:
    --------
    pd.DataFrame
        DataFrame с данными об успеваемости

    Raises:
    -------
    FileNotFoundError
        Если файл не существует
    ValueError
        Если данные имеют некорректный формат
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден")

    # Определяем параметры загрузки
    load_kwargs = {
        'encoding': 'utf-8',
        'parse_dates': ['date'],
        'infer_datetime_format': True,
    }
    load_kwargs.update(kwargs)

    try:
        df = pd.read_csv(filepath, **load_kwargs)
    except Exception as e:
        # Пробуем альтернативную кодировку
        try:
            df = pd.read_csv(filepath, encoding='cp1251', **{k: v for k, v in load_kwargs.items() if k != 'encoding'})
        except:
            raise ValueError(f"Ошибка чтения файла {filepath}: {e}")

    # Проверяем обязательные колонки
    required_columns = ['student_id', 'grade', 'subject']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

    print(f"✅ Данные успешно загружены: {len(df)} записей, {len(df.columns)} колонок")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и предобрабатывает данные об успеваемости.

    Parameters:
    -----------
    df : pd.DataFrame
        Исходный DataFrame

    Returns:
    --------
    pd.DataFrame
        Очищенный DataFrame
    """
    # Создаем копию данных
    df_clean = df.copy()

    # Удаляем дубликаты
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_count - len(df_clean)
    if duplicates_removed > 0:
        print(f"   Удалено дубликатов: {duplicates_removed}")

    # Обрабатываем пропущенные значения
    missing_before = df_clean.isnull().sum().sum()

    # Для оценок: заполняем медианной по предмету
    if 'grade' in df_clean.columns:
        subject_medians = df_clean.groupby('subject')['grade'].transform('median')
        df_clean['grade'] = df_clean['grade'].fillna(subject_medians)

    # Для посещаемости: заполняем 1 (присутствовал)
    if 'attendance' in df_clean.columns:
        df_clean['attendance'] = df_clean['attendance'].fillna(1.0)

    # Удаляем строки с пропущенными важными полями
    important_columns = ['student_id', 'grade', 'subject']
    df_clean = df_clean.dropna(subset=important_columns)

    missing_after = df_clean.isnull().sum().sum()
    if missing_before > 0:
        print(f"   Обработано пропущенных значений: {missing_before - missing_after}")

    # Фильтруем некорректные оценки (должны быть от 1 до 10)
    if 'grade' in df_clean.columns:
        initial_grades = len(df_clean)
        df_clean = df_clean[(df_clean['grade'] >= 1) & (df_clean['grade'] <= 10)]
        invalid_grades = initial_grades - len(df_clean)
        if invalid_grades > 0:
            print(f"   Удалено записей с некорректными оценками: {invalid_grades}")

    # Преобразуем типы данных
    if 'student_id' in df_clean.columns:
        df_clean['student_id'] = df_clean['student_id'].astype(str)

    if 'group' in df_clean.columns:
        df_clean['group'] = df_clean['group'].astype(str)

    # Добавляем производные поля если есть дата
    if 'date' in df_clean.columns:
        df_clean['week'] = df_clean['date'].dt.isocalendar().week
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['day_of_week'] = df_clean['date'].dt.day_name()

    print(f"✅ Данные очищены. Итоговый размер: {len(df_clean)} записей")
    return df_clean


def merge_datasets(grades_df: pd.DataFrame,
                   students_df: pd.DataFrame,
                   subjects_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Объединяет данные об оценках с информацией о студентах и предметах.

    Parameters:
    -----------
    grades_df : pd.DataFrame
        DataFrame с оценками
    students_df : pd.DataFrame
        DataFrame с информацией о студентах
    subjects_df : pd.DataFrame, optional
        DataFrame с информацией о предметах

    Returns:
    --------
    pd.DataFrame
        Объединенный DataFrame
    """
    # Проверяем наличие student_id в обоих датафреймах
    if 'student_id' not in students_df.columns:
        raise ValueError("students_df должен содержать колонку 'student_id'")

    # Объединяем оценки с информацией о студентах
    merged_df = pd.merge(
        grades_df,
        students_df,
        on='student_id',
        how='left',
        suffixes=('', '_student')
    )

    # Объединяем с информацией о предметах если предоставлена
    if subjects_df is not None and 'subject' in subjects_df.columns:
        merged_df = pd.merge(
            merged_df,
            subjects_df,
            on='subject',
            how='left',
            suffixes=('', '_subject')
        )

    # Сортируем по дате если есть
    if 'date' in merged_df.columns:
        merged_df = merged_df.sort_values('date')

    return merged_df


def generate_sample_data(num_students: int = 100,
                         num_weeks: int = 16,
                         subjects: List[str] = None) -> pd.DataFrame:
    """
    Генерирует тестовые данные об успеваемости студентов.

    Parameters:
    -----------
    num_students : int
        Количество студентов
    num_weeks : int
        Количество недель семестра
    subjects : List[str], optional
        Список предметов

    Returns:
    --------
    pd.DataFrame
        Сгенерированные тестовые данные
    """
    if subjects is None:
        subjects = ['Математика', 'Физика', 'Программирование',
                    'Английский язык', 'История', 'Философия']

    # Генерируем ID студентов
    student_ids = [f'STD{str(i).zfill(3)}' for i in range(1, num_students + 1)]

    # Генерируем группы
    groups = [f'ГРП-{i}' for i in range(1, 6)]

    # Начальная дата семестра
    start_date = datetime.now() - timedelta(weeks=num_weeks)

    data = []

    for week in range(1, num_weeks + 1):
        current_date = start_date + timedelta(weeks=week)

        for student_id in student_ids:
            # Каждый студент получает оценки по 2-4 предметам в неделю
            num_subjects = np.random.randint(2, 5)
            week_subjects = np.random.choice(subjects, num_subjects, replace=False)

            for subject in week_subjects:
                # Генерируем оценку с учетом разных факторов
                base_grade = np.random.normal(7.0, 1.5)

                # Добавляем случайность
                grade_variation = np.random.uniform(-1.5, 1.5)
                final_grade = base_grade + grade_variation

                # Ограничиваем оценку от 1 до 10 и округляем
                final_grade = max(1, min(10, round(final_grade, 1)))

                # Генерируем посещаемость (0.5 - отсутствовал наполовину, 1.0 - присутствовал)
                attendance = np.random.choice([0.5, 1.0], p=[0.1, 0.9])

                # Определяем группу студента
                group = np.random.choice(groups)

                data.append({
                    'student_id': student_id,
                    'group': group,
                    'subject': subject,
                    'grade': final_grade,
                    'attendance': attendance,
                    'date': current_date,
                    'week': week
                })

    df = pd.DataFrame(data)

    # Добавляем некоторые аномалии для реалистичности
    # 5% студентов имеют низкую успеваемость
    low_performers = np.random.choice(student_ids,
                                      size=int(0.05 * num_students),
                                      replace=False)
    df.loc[df['student_id'].isin(low_performers), 'grade'] *= 0.7

    # 10% записей имеют высокую успеваемость
    high_performers = np.random.choice(student_ids,
                                       size=int(0.1 * num_students),
                                       replace=False)
    df.loc[df['student_id'].isin(high_performers), 'grade'] = np.minimum(
        df.loc[df['student_id'].isin(high_performers), 'grade'] * 1.3, 10
    )

    print(f"✅ Сгенерировано тестовых данных: {len(df)} записей")
    print(f"   Студентов: {num_students}, Недель: {num_weeks}, Предметов: {len(subjects)}")

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Проверяет качество данных и возвращает отчет.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame для проверки

    Returns:
    --------
    Dict[str, any]
        Отчет о качестве данных
    """
    report = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'basic_stats': {}
    }

    # Проверяем пропущенные значения
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        report['missing_values'][column] = {
            'count': int(missing_count),
            'percentage': round(missing_percentage, 2)
        }

    # Определяем типы данных
    for column in df.columns:
        report['data_types'][column] = str(df[column].dtype)

    # Базовая статистика для числовых колонок
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column in df.columns:
            report['basic_stats'][column] = {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'median': float(df[column].median())
            }

    # Проверяем уникальность student_id
    if 'student_id' in df.columns:
        unique_students = df['student_id'].nunique()
        report['unique_students'] = unique_students
        report['records_per_student'] = len(df) / unique_students if unique_students > 0 else 0

    # Проверяем уникальность предметов
    if 'subject' in df.columns:
        report['unique_subjects'] = df['subject'].nunique()

    return report


def export_data(df: pd.DataFrame,
                filepath: str,
                format: str = 'csv') -> None:
    """
    Экспортирует данные в файл.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame для экспорта
    filepath : str
        Путь для сохранения файла
    format : str
        Формат файла ('csv', 'excel', 'json')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'csv':
        df.to_csv(filepath, index=False, encoding='utf-8')
    elif format.lower() == 'excel':
        df.to_excel(filepath, index=False)
    elif format.lower() == 'json':
        df.to_json(filepath, orient='records', force_ascii=False)
    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")

    print(f"✅ Данные экспортированы в {filepath}")