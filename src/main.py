#!/usr/bin/env python3
"""
Основная точка входа для запуска EduViz Dashboard
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_student_data, generate_sample_data
from src.dashboard import create_dashboard
from src.analyzer import analyze_performance
from src.utils import export_analysis_results


def main():
    parser = argparse.ArgumentParser(
        description='EduViz Dashboard - Визуализация образовательных данных'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw/grades.csv',
        help='Путь к файлу с данными'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Сгенерировать тестовые данные'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Провести анализ данных без запуска дашборда'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Порт для запуска дашборда'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Режим отладки'
    )

    args = parser.parse_args()

    # Генерация тестовых данных если нужно
    if args.generate_sample:
        print("🔧 Генерация тестовых данных...")
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)

        sample_data = generate_sample_data(num_students=100, num_weeks=16)
        sample_data.to_csv(data_dir / 'grades.csv', index=False)
        print(f"✅ Тестовые данные сохранены в {data_dir / 'grades.csv'}")
        print(f"   Сгенерировано записей: {len(sample_data)}")

    # Загрузка данных
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Файл данных не найден: {data_path}")
        print("   Используйте --generate-sample для создания тестовых данных")
        return 1

    print(f"📂 Загрузка данных из {data_path}...")
    try:
        df = load_student_data(str(data_path))
        print(f"✅ Загружено {len(df)} записей")
        print(f"   Колонки: {', '.join(df.columns)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return 1

    # Анализ данных если нужно
    if args.analyze:
        print("📊 Проведение анализа данных...")
        analysis = analyze_performance(df)

        print("\n📈 Результаты анализа:")
        print(f"   Средняя оценка: {analysis['overall']['mean_grade']:.2f}")
        print(f"   Медианная оценка: {analysis['overall']['median_grade']:.2f}")
        print(f"   Общее количество студентов: {analysis['overall']['total_students']}")
        print(f"   Количество предметов: {analysis['overall']['total_subjects']}")

        risk_students = analysis['risk_students']
        if len(risk_students) > 0:
            print(f"\n⚠️  Студенты группы риска ({len(risk_students)}):")
            for student in risk_students[:5]:  # Показываем первых 5
                print(f"   - {student['student_id']}: средняя оценка {student['avg_grade']:.2f}")

        # Экспорт результатов
        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)
        export_analysis_results(analysis, output_dir / 'analysis.json')
        print(f"\n💾 Результаты сохранены в {output_dir / 'analysis.json'}")

        if not args.debug:
            return 0

    # Запуск дашборда
    print(f"\n🚀 Запуск EduViz Dashboard на порту {args.port}...")
    print("   Откройте браузер и перейдите по адресу: http://localhost:8050")
    print("   Для остановки нажмите Ctrl+C\n")

    try:
        app = create_dashboard(df)
        app.run_server(
            host='0.0.0.0',
            port=args.port,
            debug=args.debug,
            dev_tools_hot_reload=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
    except Exception as e:
        print(f"❌ Ошибка при запуске дашборда: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())