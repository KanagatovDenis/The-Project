#!/usr/bin/env python3
"""
Скрипт для автоматической генерации отчетов
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_student_data, clean_data
from src.analyzer import analyze_performance, calculate_subject_statistics
from src.visualizer import (
    create_grade_distribution,
    create_subject_analysis,
    create_risk_students_plot
)
from src.utils import generate_report, export_analysis_results, save_visualization
import pandas as pd
import argparse
from datetime import datetime
import json


def create_html_report(report_data: dict, output_file: str) -> str:
    """
    Создает HTML отчет на основе данных.

    Parameters:
    -----------
    report_data : dict
        Данные отчета
    output_file : str
        Путь для сохранения HTML файла

    Returns:
    --------
    str
        Путь к сохраненному файлу
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет об успеваемости - {timestamp}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header .subtitle {{
                margin-top: 10px;
                opacity: 0.9;
                font-size: 1.1em;
            }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 25px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .section h2 {{
                color: #4a5568;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f7fafc;
                border-left: 4px solid #4299e1;
                padding: 20px;
                border-radius: 6px;
            }}
            .metric-card.warning {{
                border-left-color: #ed8936;
            }}
            .metric-card.danger {{
                border-left-color: #f56565;
            }}
            .metric-card.success {{
                border-left-color: #48bb78;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .table th, .table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            .table th {{
                background-color: #f7fafc;
                font-weight: 600;
                color: #4a5568;
            }}
            .table tr:hover {{
                background-color: #f8f9fa;
            }}
            .recommendation {{
                background: #fffaf0;
                border-left: 4px solid #ed8936;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .priority-high {{
                color: #c53030;
                font-weight: bold;
            }}
            .priority-medium {{
                color: #d69e2e;
            }}
            .priority-low {{
                color: #38a169;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
                color: #718096;
                font-size: 0.9em;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 600;
                margin: 0 5px;
            }}
            .badge-success {{
                background: #c6f6d5;
                color: #22543d;
            }}
            .badge-warning {{
                background: #feebc8;
                color: #744210;
            }}
            .badge-danger {{
                background: #fed7d7;
                color: #742a2a;
            }}
            .visualization-placeholder {{
                background: #f7fafc;
                border: 2px dashed #cbd5e0;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                color: #a0aec0;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Отчет об успеваемости</h1>
            <div class="subtitle">
                Сгенерировано: {timestamp} | {report_data['metadata'].get('report_type', 'Еженедельный').title()} отчет
            </div>
        </div>

        <div class="section">
            <h2>📈 Сводные показатели</h2>
            <div class="metrics-grid">
    """

    # Добавляем метрики
    summary = report_data.get('summary', {})

    metrics = [
        ('total_students', '👥 Студентов', ''),
        ('average_grade', '⭐ Средняя оценка', f"{summary.get('average_grade', 0):.2f}"),
        ('pass_rate', '✅ Успеваемость', f"{summary.get('pass_rate', 0):.1f}%"),
        ('risk_students_count', '⚠️  Группа риска', f"{summary.get('risk_students_count', 0)}"),
        ('risk_percentage', '📊 % риска', f"{summary.get('risk_percentage', 0):.1f}%"),
        ('total_subjects', '📚 Предметов', f"{summary.get('total_subjects', 0)}")
    ]

    for key, label, value in metrics:
        value_display = value if value else summary.get(key, 0)

        # Определяем класс для карточки
        card_class = ""
        if key == 'risk_students_count' and summary.get('risk_students_count', 0) > 5:
            card_class = "warning"
        elif key == 'average_grade' and summary.get('average_grade', 0) < 5:
            card_class = "danger"
        elif key == 'pass_rate' and summary.get('pass_rate', 0) > 80:
            card_class = "success"

        html_content += f"""
                <div class="metric-card {card_class}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value_display}</div>
                </div>
        """

    html_content += """
            </div>
        </div>

        <div class="section">
            <h2>🏆 Топ студентов</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Студент</th>
                        <th>Средняя оценка</th>
                        <th>Предметы</th>
                        <th>Статус</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Добавляем топ студентов
    top_students = report_data.get('details', {}).get('top_students', [])
    for student in top_students[:10]:
        grade = student.get('average_grade', 0)

        if grade >= 9:
            status_badge = '<span class="badge badge-success">Отлично</span>'
        elif grade >= 7:
            status_badge = '<span class="badge badge-success">Хорошо</span>'
        elif grade >= 5:
            status_badge = '<span class="badge badge-warning">Удовл.</span>'
        else:
            status_badge = '<span class="badge badge-danger">Риск</span>'

        html_content += f"""
                    <tr>
                        <td>{student.get('student_id', 'N/A')}</td>
                        <td>{grade:.2f}</td>
                        <td>{student.get('subject_count', 0)}</td>
                        <td>{status_badge}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>📚 Топ предметов</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Предмет</th>
                        <th>Средняя оценка</th>
                        <th>Студентов</th>
                        <th>Успеваемость</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Добавляем топ предметов
    top_subjects = report_data.get('details', {}).get('top_subjects', [])
    for subject in top_subjects[:10]:
        pass_rate = min(100, max(0, subject.get('average_grade', 0) * 10))

        html_content += f"""
                    <tr>
                        <td>{subject.get('subject', 'N/A')}</td>
                        <td>{subject.get('average_grade', 0):.2f}</td>
                        <td>{subject.get('student_count', 0)}</td>
                        <td>{pass_rate:.1f}%</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>⚠️  Анализ рисков</h2>
            <p>Студентов в группе риска: <strong>{risk_count}</strong></p>
    """.format(risk_count=len(report_data.get('details', {}).get('risk_analysis', [])))

    # Добавляем анализ рисков
    risk_students = report_data.get('details', {}).get('risk_analysis', [])
    if risk_students:
        html_content += """
            <table class="table">
                <thead>
                    <tr>
                        <th>Студент</th>
                        <th>Средняя оценка</th>
                        <th>Факторы риска</th>
                        <th>Уровень риска</th>
                    </tr>
                </thead>
                <tbody>
        """

        for student in risk_students[:5]:
            risk_level = student.get('risk_level', 'medium')
            risk_class = "badge-warning"
            if risk_level == 'high':
                risk_class = "badge-danger"
            elif risk_level == 'low':
                risk_class = "badge-success"

            html_content += f"""
                    <tr>
                        <td>{student.get('student_id', 'N/A')}</td>
                        <td>{student.get('avg_grade', 0):.2f}</td>
                        <td>{len(student.get('risk_factors', [])) if isinstance(student.get('risk_factors'), list) else 0}</td>
                        <td><span class="badge {risk_class}">{risk_level}</span></td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>
        """
    else:
        html_content += """
            <p style="color: #48bb78;">✅ Нет студентов в группе риска</p>
        """

    html_content += """
        </div>

        <div class="section">
            <h2>💡 Рекомендации</h2>
    """

    # Добавляем рекомендации
    recommendations = report_data.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            priority_class = f"priority-{rec.get('priority', 'medium')}"

            html_content += f"""
            <div class="recommendation">
                <div class="{priority_class}">🔸 {rec.get('description', '')}</div>
                <p><strong>Действие:</strong> {rec.get('action', '')}</p>
            </div>
            """
    else:
        html_content += """
            <p>Все показатели в норме. Продолжайте в том же духе! 🎉</p>
        """

    html_content += f"""
        </div>

        <div class="section">
            <h2>📊 Визуализации</h2>
            <div class="visualization-placeholder">
                <p>📈 Графики и диаграммы будут отображены здесь</p>
                <p><small>Для просмотра интерактивных визуализаций используйте EduViz Dashboard</small></p>
            </div>
        </div>

        <div class="footer">
            <p>📄 Отчет сгенерирован автоматически системой EduViz Dashboard</p>
            <p>📧 По вопросам обращайтесь: eduviz@example.com</p>
            <p>© 2024 EduViz Dashboard | Все права защищены</p>
        </div>
    </body>
    </html>
    """

    # Сохраняем файл
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML отчет сохранен: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Генератор отчетов для EduViz Dashboard')

    parser.add_argument('--data', '-d',
                        default='data/raw/grades.csv',
                        help='Путь к данным (по умолчанию: data/raw/grades.csv)')

    parser.add_argument('--type', '-t',
                        choices=['weekly', 'monthly', 'detailed', 'full'],
                        default='weekly',
                        help='Тип отчета')

    parser.add_argument('--output-dir', '-o',
                        default='reports',
                        help='Директория для сохранения отчетов')

    parser.add_argument('--visualizations', '-v',
                        action='store_true',
                        help='Генерировать визуализации')

    parser.add_argument('--email',
                        help='Email для отправки отчета (опционально)')

    args = parser.parse_args()

    print("📊 Запуск генератора отчетов...")
    print(f"   Тип отчета: {args.type}")
    print(f"   Источник данных: {args.data}")
    print(f"   Выходная директория: {args.output_dir}")

    # Проверяем наличие данных
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Файл данных не найден: {args.data}")
        print("   Используйте scripts/generate_sample_data.py для создания тестовых данных")
        return 1

    # Загружаем данные
    print("📂 Загрузка данных...")
    try:
        df = load_student_data(str(data_path))
        df_clean = clean_data(df)
        print(f"✅ Загружено {len(df_clean)} записей")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return 1

    # Генерируем отчет
    print("📈 Генерация отчета...")
    report_data = generate_report(df_clean, report_type=args.type)

    # Создаем временную метку
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Сохраняем JSON отчет
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / f"report_{args.type}_{timestamp}.json"
    export_analysis_results(report_data, str(json_file))

    # Создаем HTML отчет
    html_file = output_dir / f"report_{args.type}_{timestamp}.html"
    create_html_report(report_data, str(html_file))

    # Генерируем визуализации если нужно
    if args.visualizations:
        print("🎨 Генерация визуализаций...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        try:
            # Создаем основные визуализации
            from src.visualizer import (
                create_grade_distribution,
                create_subject_analysis,
                create_risk_students_plot
            )

            # Распределение оценок
            grade_fig = create_grade_distribution(df_clean)
            save_visualization(grade_fig, str(viz_dir / f"grade_distribution_{timestamp}.html"))

            # Анализ предметов
            subject_fig = create_subject_analysis(df_clean)
            save_visualization(subject_fig, str(viz_dir / f"subject_analysis_{timestamp}.html"))

            # Студенты группы риска
            risk_fig = create_risk_students_plot(df_clean)
            save_visualization(risk_fig, str(viz_dir / f"risk_students_{timestamp}.html"))

            print(f"✅ Визуализации сохранены в {viz_dir}")

        except Exception as e:
            print(f"⚠️  Ошибка генерации визуализаций: {e}")

    # Выводим сводку
    print(f"\n✅ Отчет успешно сгенерирован!")
    print(f"   JSON отчет: {json_file}")
    print(f"   HTML отчет: {html_file}")

    summary = report_data.get('summary', {})
    print(f"\n📊 Ключевые показатели:")
    print(f"   Студентов: {summary.get('total_students', 0)}")
    print(f"   Средняя оценка: {summary.get('average_grade', 0):.2f}")
    print(f"   Успеваемость: {summary.get('pass_rate', 0):.1f}%")
    print(f"   Студентов в группе риска: {summary.get('risk_students_count', 0)}")

    # Отправляем email если указан
    if args.email:
        print(f"\n📧 Отправка отчета на {args.email}...")
        # Здесь можно добавить логику отправки email
        print("   (Функция отправки email в разработке)")

    print(f"\n🚀 Отчет готов к использованию!")
    print(f"   Откройте {html_file} в браузере для просмотра")

    return 0


if __name__ == '__main__':
    sys.exit(main())