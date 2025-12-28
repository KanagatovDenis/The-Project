"""
Вспомогательные функции для проекта
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def export_to_html(fig, filename: str, title: str = "Визуализация") -> str:
    """
    Экспортирует график Plotly в HTML файл с улучшенным форматированием.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        График для экспорта
    filename : str
        Имя файла для сохранения
    title : str
        Заголовок HTML страницы

    Returns:
    --------
    str
        Путь к сохраненному файлу
    """
    import plotly.io as pio

    # Создаем полный HTML контент
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e0e0e0;
            }}
            .info {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 12px;
                color: #666;
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{title}</h1>
                <p>Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="info">
                <strong>Информация:</strong> Эта визуализация была автоматически сгенерирована системой EduViz Dashboard.
            </div>

            <div id="plotly-chart"></div>

            <div class="footer">
                <p>EduViz Dashboard © 2024 | Система визуализации образовательных данных</p>
            </div>
        </div>

        <script>
            var graph = {fig.to_json()};
            Plotly.newPlot('plotly-chart', graph.data, graph.layout, {{responsive: true}});

            // Добавляем обработчик изменения размера окна
            window.addEventListener('resize', function() {{
                Plotly.Plots.resize(document.getElementById('plotly-chart'));
            }});
        </script>
    </body>
    </html>
    """

    # Сохраняем файл
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML файл сохранен: {filepath}")
    return str(filepath)


def export_analysis_results(results: Dict[str, Any], filename: str) -> str:
    """
    Экспортирует результаты анализа в JSON файл.

    Parameters:
    -----------
    results : Dict[str, Any]
        Результаты анализа
    filename : str
        Имя файла для сохранения

    Returns:
    --------
    str
        Путь к сохраненному файлу
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Добавляем метаданные
    results_with_metadata = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'tool': 'EduViz Dashboard',
            'version': '1.0.0'
        },
        'data': results
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ Результаты анализа сохранены: {filepath}")
    return str(filepath)


def save_visualization(fig, filepath: str, format: str = 'html',
                       width: int = 1200, height: int = 600) -> str:
    """
    Сохраняет визуализацию в указанном формате.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        График для сохранения
    filepath : str
        Путь для сохранения файла
    format : str
        Формат файла ('html', 'png', 'pdf', 'svg')
    width : int
        Ширина изображения (для растровых форматов)
    height : int
        Высота изображения (для растровых форматов)

    Returns:
    --------
    str
        Путь к сохраненному файлу
    """
    import plotly.io as pio

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    format = format.lower()

    if format == 'html':
        # Используем нашу улучшенную функцию экспорта
        return export_to_html(fig, filepath, title="Визуализация данных")

    elif format == 'png':
        fig.write_image(str(filepath), width=width, height=height)
        print(f"✅ PNG файл сохранен: {filepath}")

    elif format == 'pdf':
        fig.write_image(str(filepath), width=width, height=height)
        print(f"✅ PDF файл сохранен: {filepath}")

    elif format == 'svg':
        fig.write_image(str(filepath), width=width, height=height)
        print(f"✅ SVG файл сохранен: {filepath}")

    else:
        raise ValueError(f"Неподдерживаемый формат: {format}")

    return str(filepath)


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Вычисляет основные статистические показатели.

    Parameters:
    -----------
    data : List[float]
        Список числовых значений

    Returns:
    --------
    Dict[str, float]
        Статистические показатели
    """
    if not data:
        return {}

    arr = np.array(data)

    stats = {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        'skewness': float(pd.Series(arr).skew()),
        'count': int(len(arr))
    }

    return stats


def format_number(value: float, decimals: int = 2) -> str:
    """
    Форматирует число для отображения.

    Parameters:
    -----------
    value : float
        Число для форматирования
    decimals : int
        Количество знаков после запятой

    Returns:
    --------
    str
        Отформатированная строка
    """
    if value is None:
        return "N/A"

    try:
        if abs(value) >= 1000000:
            return f"{value / 1000000:.{decimals}f}M"
        elif abs(value) >= 1000:
            return f"{value / 1000:.{decimals}f}K"
        elif abs(value) < 0.001 and value != 0:
            return f"{value:.{decimals}e}"
        else:
            return f"{value:.{decimals}f}"
    except:
        return str(value)


def generate_report(df: pd.DataFrame, report_type: str = 'weekly') -> Dict[str, Any]:
    """
    Генерирует отчет на основе данных.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными
    report_type : str
        Тип отчета ('weekly', 'monthly', 'detailed')

    Returns:
    --------
    Dict[str, Any]
        Сгенерированный отчет
    """
    from .analyzer import analyze_performance, calculate_subject_statistics

    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': report_type,
            'data_source': 'EduViz Dashboard',
            'period': None
        },
        'summary': {},
        'details': {},
        'recommendations': []
    }

    # Определяем период отчета
    if 'date' in df.columns:
        min_date = df['date'].min()
        max_date = df['date'].max()
        report['metadata']['period'] = {
            'start': min_date.strftime('%Y-%m-%d') if hasattr(min_date, 'strftime') else str(min_date),
            'end': max_date.strftime('%Y-%m-%d') if hasattr(max_date, 'strftime') else str(max_date)
        }

    # Анализируем данные
    analysis = analyze_performance(df)
    subject_stats = calculate_subject_statistics(df)

    # Сводная информация
    overall = analysis['overall']
    report['summary'] = {
        'total_students': overall['total_students'],
        'total_subjects': overall['total_subjects'],
        'average_grade': overall['mean_grade'],
        'median_grade': overall['median_grade'],
        'pass_rate': round((df['grade'] >= 5).mean() * 100, 1),
        'risk_students_count': len(analysis['risk_students']),
        'risk_percentage': round(len(analysis['risk_students']) / overall['total_students'] * 100, 1)
    }

    # Детальная информация
    report['details'] = {
        'top_subjects': [],
        'top_students': [],
        'risk_analysis': analysis['risk_students'][:10]  # Первые 10 студентов группы риска
    }

    # Топ предметов по средней оценке
    subject_means = df.groupby('subject')['grade'].mean().sort_values(ascending=False)
    for subject, mean_grade in subject_means.head(5).items():
        report['details']['top_subjects'].append({
            'subject': subject,
            'average_grade': round(mean_grade, 2),
            'student_count': df[df['subject'] == subject]['student_id'].nunique()
        })

    # Топ студентов
    student_means = df.groupby('student_id')['grade'].mean().sort_values(ascending=False)
    for student, mean_grade in student_means.head(5).items():
        report['details']['top_students'].append({
            'student_id': student,
            'average_grade': round(mean_grade, 2),
            'subject_count': df[df['student_id'] == student]['subject'].nunique()
        })

    # Рекомендации
    if analysis['risk_students']:
        report['recommendations'].append({
            'type': 'risk_mitigation',
            'priority': 'high',
            'description': f'Необходимо уделить внимание {len(analysis["risk_students"])} студентам группы риска',
            'action': 'Провести индивидуальные консультации с кураторами групп'
        })

    # Анализ предметов с низкой успеваемостью
    low_perf_subjects = [s for s, stats in subject_stats.items() if stats['basic']['mean'] < 5]
    if low_perf_subjects:
        report['recommendations'].append({
            'type': 'curriculum',
            'priority': 'medium',
            'description': f'Низкая успеваемость по предметам: {", ".join(low_perf_subjects[:3])}',
            'action': 'Пересмотреть методику преподавания по данным предметам'
        })

    # Анализ посещаемости если есть данные
    if 'attendance' in df.columns:
        avg_attendance = df['attendance'].mean()
        if avg_attendance < 0.8:
            report['recommendations'].append({
                'type': 'attendance',
                'priority': 'medium',
                'description': f'Средняя посещаемость составляет {avg_attendance * 100:.1f}%',
                'action': 'Внедрить систему мониторинга посещаемости'
            })

    return report


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Проверяет корректность конфигурации.

    Parameters:
    -----------
    config : Dict[str, Any]
        Конфигурационный словарь

    Returns:
    --------
    bool
        True если конфигурация корректна
    """
    required_fields = ['data_source', 'analysis_period', 'risk_threshold']

    for field in required_fields:
        if field not in config:
            print(f"❌ Отсутствует обязательное поле: {field}")
            return False

    # Проверяем значения
    if config['risk_threshold'] < 1 or config['risk_threshold'] > 10:
        print(f"❌ Некорректное значение risk_threshold: {config['risk_threshold']}")
        return False

    return True


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла.

    Parameters:
    -----------
    filepath : str
        Путь к файлу конфигурации

    Returns:
    --------
    Dict[str, Any]
        Загруженная конфигурация
    """
    path = Path(filepath)

    if not path.exists():
        print(f"⚠️  Файл конфигурации не найден: {filepath}")
        return {}

    try:
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            print(f"❌ Неподдерживаемый формат файла: {path.suffix}")
            return {}

        if validate_config(config):
            print(f"✅ Конфигурация загружена из {filepath}")
            return config
        else:
            print(f"❌ Некорректная конфигурация в файле {filepath}")
            return {}

    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return {}


def save_config(config: Dict[str, Any], filepath: str) -> bool:
    """
    Сохраняет конфигурацию в файл.

    Parameters:
    -----------
    config : Dict[str, Any]
        Конфигурационный словарь
    filepath : str
        Путь для сохранения файла

    Returns:
    --------
    bool
        True если сохранение успешно
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
        else:
            print(f"❌ Неподдерживаемый формат файла: {path.suffix}")
            return False

        print(f"✅ Конфигурация сохранена в {filepath}")
        return True

    except Exception as e:
        print(f"❌ Ошибка сохранения конфигурации: {e}")
        return False


def create_sample_config() -> Dict[str, Any]:
    """
    Создает образец конфигурации.

    Returns:
    --------
    Dict[str, Any]
        Образец конфигурации
    """
    config = {
        'data_source': {
            'type': 'csv',
            'path': 'data/raw/grades.csv',
            'encoding': 'utf-8'
        },
        'analysis_period': {
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        },
        'risk_threshold': 5.0,
        'min_records_per_student': 3,
        'visualization': {
            'color_scheme': 'plotly',
            'font_size': 14,
            'default_width': 1200,
            'default_height': 600
        },
        'reporting': {
            'weekly_report': True,
            'auto_generate': True,
            'export_formats': ['html', 'pdf']
        },
        'notifications': {
            'enabled': True,
            'email_alerts': False,
            'risk_student_threshold': 3
        }
    }

    return config


def cleanup_temp_files(directory: str = 'temp',
                       max_age_days: int = 7) -> int:
    """
    Удаляет временные файлы старше указанного возраста.

    Parameters:
    -----------
    directory : str
        Директория для очистки
    max_age_days : int
        Максимальный возраст файлов в днях

    Returns:
    --------
    int
        Количество удаленных файлов
    """
    temp_dir = Path(directory)

    if not temp_dir.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    deleted_count = 0

    for file_path in temp_dir.glob('*'):
        if file_path.is_file():
            try:
                # Получаем время модификации файла
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    print(f"🗑️  Удален старый файл: {file_path}")
            except Exception as e:
                print(f"⚠️  Не удалось удалить файл {file_path}: {e}")

    print(f"✅ Очистка завершена. Удалено файлов: {deleted_count}")
    return deleted_count