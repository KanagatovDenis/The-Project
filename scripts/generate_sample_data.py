#!/usr/bin/env python3
"""
Скрипт для генерации тестовых данных
"""

import sys
from pathlib import Path

# Добавляем путь к src в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import generate_sample_data
import pandas as pd
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Генератор тестовых данных для EduViz Dashboard')

    parser.add_argument('--output', '-o',
                        default='data/raw/grades.csv',
                        help='Путь для сохранения данных (по умолчанию: data/raw/grades.csv)')

    parser.add_argument('--students', '-s',
                        type=int,
                        default=100,
                        help='Количество студентов (по умолчанию: 100)')

    parser.add_argument('--weeks', '-w',
                        type=int,
                        default=16,
                        help='Количество недель семестра (по умолчанию: 16)')

    parser.add_argument('--subjects',
                        nargs='+',
                        default=['Математика', 'Физика', 'Программирование',
                                 'Английский язык', 'История', 'Философия'],
                        help='Список предметов')

    parser.add_argument('--format', '-f',
                        choices=['csv', 'excel', 'json'],
                        default='csv',
                        help='Формат выходного файла')

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Подробный вывод')

    args = parser.parse_args()

    print("🚀 Запуск генератора тестовых данных...")
    print(f"   Количество студентов: {args.students}")
    print(f"   Количество недель: {args.weeks}")
    print(f"   Предметы: {', '.join(args.subjects)}")
    print(f"   Выходной файл: {args.output}")
    print(f"   Формат: {args.format}")

    # Генерируем данные
    df = generate_sample_data(
        num_students=args.students,
        num_weeks=args.weeks,
        subjects=args.subjects
    )

    # Создаем директорию если не существует
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем в указанном формате
    if args.format == 'csv':
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif args.format == 'excel':
        df.to_excel(output_path, index=False)
    elif args.format == 'json':
        df.to_json(output_path, orient='records', force_ascii=False)

    # Выводим статистику
    print(f"\n✅ Данные успешно сгенерированы!")
    print(f"   Всего записей: {len(df):,}")
    print(f"   Уникальных студентов: {df['student_id'].nunique()}")
    print(f"   Уникальных предметов: {df['subject'].nunique()}")
    print(f"   Диапазон оценок: {df['grade'].min():.1f} - {df['grade'].max():.1f}")
    print(f"   Средняя оценка: {df['grade'].mean():.2f}")

    if 'group' in df.columns:
        print(f"   Групп: {df['group'].nunique()}")

    if args.verbose:
        print("\n📊 Пример данных:")
        print(df.head())

        print("\n📈 Статистика по предметам:")
        subject_stats = df.groupby('subject').agg({
            'grade': ['mean', 'count'],
            'student_id': 'nunique'
        }).round(2)
        print(subject_stats)

    print(f"\n💾 Файл сохранен: {output_path.absolute()}")
    print("   Для использования в дашборде выполните: python src/main.py")


if __name__ == '__main__':
    main()