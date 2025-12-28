"""
Аналитический модуль для обработки образовательных данных
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def analyze_performance(df: pd.DataFrame,
                        risk_threshold: float = 5.0,
                        min_records: int = 3) -> Dict[str, Any]:
    """
    Проводит комплексный анализ успеваемости.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    risk_threshold : float
        Порог для определения студентов группы риска
    min_records : int
        Минимальное количество записей для анализа студента

    Returns:
    --------
    Dict[str, Any]
        Результаты анализа
    """
    results = {
        'overall': {},
        'by_subject': {},
        'by_group': {},
        'risk_students': [],
        'trends': {},
        'timestamp': datetime.now().isoformat()
    }

    # 1. Общая статистика
    results['overall'] = {
        'total_records': int(len(df)),
        'total_students': int(df['student_id'].nunique()),
        'total_subjects': int(df['subject'].nunique()),
        'mean_grade': float(df['grade'].mean()),
        'median_grade': float(df['grade'].median()),
        'std_grade': float(df['grade'].std()),
        'min_grade': float(df['grade'].min()),
        'max_grade': float(df['grade'].max()),
        'grade_distribution': df['grade'].value_counts().sort_index().to_dict()
    }

    # Добавляем статистику по группам если есть
    if 'group' in df.columns:
        results['overall']['total_groups'] = int(df['group'].nunique())

    # 2. Статистика по предметам
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        subject_stats = {
            'mean_grade': float(subject_data['grade'].mean()),
            'median_grade': float(subject_data['grade'].median()),
            'std_grade': float(subject_data['grade'].std()),
            'student_count': int(subject_data['student_id'].nunique()),
            'record_count': int(len(subject_data)),
            'pass_rate': float((subject_data['grade'] >= 4).mean() * 100),
            'excellent_rate': float((subject_data['grade'] >= 9).mean() * 100)
        }

        # Добавляем распределение оценок
        grade_counts = subject_data['grade'].value_counts().sort_index()
        subject_stats['grade_distribution'] = {
            str(k): int(v) for k, v in grade_counts.items()
        }

        results['by_subject'][subject] = subject_stats

    # 3. Статистика по группам
    if 'group' in df.columns:
        for group in df['group'].unique():
            group_data = df[df['group'] == group]
            group_stats = {
                'mean_grade': float(group_data['grade'].mean()),
                'median_grade': float(group_data['grade'].median()),
                'student_count': int(group_data['student_id'].nunique()),
                'subject_count': int(group_data['subject'].nunique()),
                'pass_rate': float((group_data['grade'] >= 4).mean() * 100),
                'attendance_rate': None
            }

            # Добавляем посещаемость если есть
            if 'attendance' in group_data.columns:
                group_stats['attendance_rate'] = float(group_data['attendance'].mean() * 100)

            results['by_group'][group] = group_stats

    # 4. Студенты группы риска
    student_stats = df.groupby('student_id').agg({
        'grade': ['mean', 'count', 'std'],
        'subject': 'nunique'
    }).round(2)

    student_stats.columns = ['avg_grade', 'grade_count', 'grade_std', 'subject_count']
    student_stats = student_stats.reset_index()

    # Фильтруем студентов с достаточным количеством записей
    student_stats = student_stats[student_stats['grade_count'] >= min_records]

    # Определяем студентов группы риска
    risk_students = student_stats[student_stats['avg_grade'] < risk_threshold]

    for _, student in risk_students.iterrows():
        student_info = {
            'student_id': student['student_id'],
            'avg_grade': float(student['avg_grade']),
            'grade_count': int(student['grade_count']),
            'grade_std': float(student['grade_std']),
            'subject_count': int(student['subject_count']),
            'risk_level': 'high' if student['avg_grade'] < 4.0 else 'medium'
        }

        # Добавляем информацию о группе если есть
        if 'group' in df.columns:
            student_groups = df[df['student_id'] == student['student_id']]['group'].unique()
            student_info['groups'] = list(student_groups)

        # Анализ тренда успеваемости
        if 'week' in df.columns or 'date' in df.columns:
            student_data = df[df['student_id'] == student['student_id']]

            if 'week' not in student_data.columns and 'date' in student_data.columns:
                student_data['week'] = student_data['date'].dt.isocalendar().week

            if 'week' in student_data.columns:
                weekly_grades = student_data.groupby('week')['grade'].mean()
                if len(weekly_grades) > 1:
                    # Вычисляем тренд (положительный/отрицательный)
                    x = np.arange(len(weekly_grades))
                    y = weekly_grades.values
                    slope = np.polyfit(x, y, 1)[0]
                    student_info['trend'] = 'positive' if slope > 0.1 else 'negative' if slope < -0.1 else 'stable'
                    student_info['trend_slope'] = float(slope)

        results['risk_students'].append(student_info)

    # 5. Анализ трендов
    if 'week' in df.columns or 'date' in df.columns:
        if 'week' not in df.columns:
            df['week'] = df['date'].dt.isocalendar().week

        # Тренд средней успеваемости по неделям
        weekly_trend = df.groupby('week')['grade'].mean().reset_index()

        results['trends']['weekly'] = {
            'weeks': weekly_trend['week'].tolist(),
            'mean_grades': weekly_trend['grade'].round(2).tolist()
        }

        # Вычисляем общий тренд
        if len(weekly_trend) > 1:
            x = np.arange(len(weekly_trend))
            y = weekly_trend['grade'].values
            slope, intercept = np.polyfit(x, y, 1)

            results['trends']['overall_trend'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'direction': 'improving' if slope > 0.05 else 'declining' if slope < -0.05 else 'stable',
                'prediction_next_week': float(slope * len(weekly_trend) + intercept)
            }

    # 6. Корреляционный анализ
    try:
        # Создаем сводную таблицу для корреляции
        pivot_data = df.pivot_table(
            index='student_id',
            columns='subject',
            values='grade',
            aggfunc='mean'
        ).dropna(thresh=3)  # Удаляем студентов с менее чем 3 предметами

        if len(pivot_data.columns) > 1 and len(pivot_data) > 5:
            corr_matrix = pivot_data.corr()

            # Находим наиболее коррелированные пары предметов
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:  # Значимая корреляция
                        correlations.append({
                            'subject1': corr_matrix.columns[i],
                            'subject2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate' if abs(
                                corr_value) > 0.5 else 'weak'
                        })

            # Сортируем по абсолютному значению корреляции
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            results['correlations'] = correlations[:10]  # Топ-10 корреляций
    except Exception as e:
        results['correlations'] = f"Не удалось вычислить корреляции: {str(e)}"

    # 7. Кластеризация студентов
    try:
        if len(student_stats) >= 10:
            # Подготавливаем данные для кластеризации
            cluster_features = student_stats[['avg_grade', 'grade_count', 'grade_std']].fillna(0)

            # Масштабируем данные
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_features)

            # Применяем K-means
            n_clusters = min(4, len(student_stats) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)

            student_stats['cluster'] = clusters

            # Анализируем кластеры
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_data = student_stats[student_stats['cluster'] == cluster_id]
                cluster_info = {
                    'student_count': int(len(cluster_data)),
                    'avg_grade_mean': float(cluster_data['avg_grade'].mean()),
                    'avg_grade_std': float(cluster_data['avg_grade'].std()),
                    'student_ids': cluster_data['student_id'].tolist()[:10]  # Первые 10 ID
                }
                cluster_analysis[f'cluster_{cluster_id}'] = cluster_info

            results['clusters'] = cluster_analysis
    except Exception as e:
        results['clusters'] = f"Не удалось выполнить кластеризацию: {str(e)}"

    return results


def identify_at_risk_students(df: pd.DataFrame,
                              threshold: float = 5.0,
                              min_weeks: int = 3,
                              decline_threshold: float = 1.5) -> pd.DataFrame:
    """
    Идентифицирует студентов группы риска на основе нескольких критериев.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    threshold : float
        Порог средней оценки
    min_weeks : int
        Минимальное количество недель для анализа тренда
    decline_threshold : float
        Порог снижения успеваемости

    Returns:
    --------
    pd.DataFrame
        DataFrame с информацией о студентах группы риска
    """
    # Создаем копию данных
    df_copy = df.copy()

    # Добавляем неделю если есть дата
    if 'week' not in df_copy.columns and 'date' in df_copy.columns:
        df_copy['week'] = df_copy['date'].dt.isocalendar().week

    results = []

    for student_id in df_copy['student_id'].unique():
        student_data = df_copy[df_copy['student_id'] == student_id]

        # Базовые метрики
        avg_grade = student_data['grade'].mean()
        grade_std = student_data['grade'].std()
        subject_count = student_data['subject'].nunique()
        total_grades = len(student_data)

        # Проверяем критерии риска
        risk_factors = []

        # 1. Низкая средняя оценка
        if avg_grade < threshold:
            risk_factors.append(f"Низкая средняя оценка ({avg_grade:.2f})")

        # 2. Высокая изменчивость оценок
        if grade_std > 2.5:
            risk_factors.append(f"Высокая изменчивость оценок (σ={grade_std:.2f})")

        # 3. Анализ тренда если есть данные по неделям
        if 'week' in student_data.columns and student_data['week'].nunique() >= min_weeks:
            weekly_grades = student_data.groupby('week')['grade'].mean().sort_index()

            if len(weekly_grades) >= min_weeks:
                # Вычисляем тренд
                x = np.arange(len(weekly_grades))
                y = weekly_grades.values
                slope = np.polyfit(x, y, 1)[0]

                # Проверяем снижение успеваемости
                if slope < -0.2:
                    risk_factors.append(f"Снижение успеваемости (тренд: {slope:.2f}/неделю)")

                # Проверяем резкое падение
                first_half = np.mean(y[:len(y) // 2])
                second_half = np.mean(y[len(y) // 2:])

                if first_half - second_half > decline_threshold:
                    risk_factors.append(f"Резкое падение успеваемости ({first_half:.1f} → {second_half:.1f})")

        # 4. Низкая посещаемость если есть данные
        if 'attendance' in student_data.columns:
            attendance_rate = student_data['attendance'].mean()
            if attendance_rate < 0.7:
                risk_factors.append(f"Низкая посещаемость ({attendance_rate * 100:.1f}%)")

        # Если есть факторы риска, добавляем студента в результаты
        if risk_factors:
            student_info = {
                'student_id': student_id,
                'avg_grade': avg_grade,
                'grade_std': grade_std,
                'subject_count': subject_count,
                'total_grades': total_grades,
                'risk_factors': risk_factors,
                'risk_score': len(risk_factors),
                'recommendations': generate_recommendations(risk_factors)
            }

            # Добавляем информацию о группе если есть
            if 'group' in student_data.columns:
                student_info['group'] = student_data['group'].iloc[0]

            results.append(student_info)

    # Создаем DataFrame и сортируем по оценке риска
    if results:
        risk_df = pd.DataFrame(results)
        risk_df = risk_df.sort_values(['risk_score', 'avg_grade'], ascending=[False, True])
        return risk_df
    else:
        return pd.DataFrame()  # Пустой DataFrame если нет студентов группы риска


def calculate_subject_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Вычисляет подробную статистику по каждому предмету.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Статистика по предметам
    """
    subject_stats = {}

    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]

        # Базовые статистики
        grades = subject_data['grade']

        stats = {
            'basic': {
                'mean': float(grades.mean()),
                'median': float(grades.median()),
                'std': float(grades.std()),
                'min': float(grades.min()),
                'max': float(grades.max()),
                'range': float(grades.max() - grades.min()),
                'iqr': float(grades.quantile(0.75) - grades.quantile(0.25))
            },
            'counts': {
                'total_students': int(subject_data['student_id'].nunique()),
                'total_records': int(len(subject_data)),
                'unique_weeks': int(subject_data['week'].nunique()) if 'week' in subject_data.columns else None
            },
            'distribution': {
                'excellent_count': int((grades >= 9).sum()),
                'good_count': int(((grades >= 7) & (grades < 9)).sum()),
                'satisfactory_count': int(((grades >= 5) & (grades < 7)).sum()),
                'fail_count': int((grades < 5).sum()),
                'excellent_percentage': float((grades >= 9).mean() * 100),
                'pass_percentage': float((grades >= 5).mean() * 100)
            },
            'percentiles': {
                'p10': float(grades.quantile(0.10)),
                'p25': float(grades.quantile(0.25)),
                'p50': float(grades.quantile(0.50)),
                'p75': float(grades.quantile(0.75)),
                'p90': float(grades.quantile(0.90))
            }
        }

        # Анализ по группам если есть
        if 'group' in subject_data.columns:
            group_stats = {}
            for group in subject_data['group'].unique():
                group_data = subject_data[subject_data['group'] == group]
                group_grades = group_data['grade']

                group_stats[group] = {
                    'mean': float(group_grades.mean()),
                    'count': int(len(group_data)),
                    'std': float(group_grades.std()),
                    'pass_rate': float((group_grades >= 5).mean() * 100)
                }

            stats['by_group'] = group_stats

        # Анализ тренда если есть недели
        if 'week' in subject_data.columns and subject_data['week'].nunique() > 1:
            weekly_avg = subject_data.groupby('week')['grade'].mean()

            stats['trend'] = {
                'weekly_means': weekly_avg.round(2).to_dict(),
                'trend_slope': float(np.polyfit(range(len(weekly_avg)), weekly_avg.values, 1)[0]) if len(
                    weekly_avg) > 1 else None,
                'volatility': float(weekly_avg.std())
            }

        subject_stats[subject] = stats

    return subject_stats


def predict_final_grades(df: pd.DataFrame,
                         current_week: Optional[int] = None) -> pd.DataFrame:
    """
    Прогнозирует итоговые оценки студентов на основе текущей успеваемости.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    current_week : int, optional
        Текущая неделя семестра

    Returns:
    --------
    pd.DataFrame
        Прогноз итоговых оценок
    """
    # Определяем текущую неделю
    if current_week is None:
        if 'week' in df.columns:
            current_week = df['week'].max()
        else:
            current_week = 10  # Значение по умолчанию

    predictions = []

    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]

        for subject in student_data['subject'].unique():
            subject_data = student_data[student_data['subject'] == subject]

            if 'week' in subject_data.columns:
                # Получаем оценки по неделям
                weekly_grades = subject_data.groupby('week')['grade'].mean().sort_index()

                if len(weekly_grades) >= 2:
                    # Прогнозируем с помощью линейной регрессии
                    weeks = np.array(weekly_grades.index)
                    grades = weekly_grades.values

                    # Линейная регрессия
                    slope, intercept = np.polyfit(weeks, grades, 1)

                    # Прогноз на конец семестра (16 недель)
                    final_grade_pred = slope * 16 + intercept
                    final_grade_pred = max(1, min(10, final_grade_pred))  # Ограничиваем от 1 до 10

                    # Текущая средняя оценка
                    current_avg = grades.mean()

                    # Уверенность прогноза
                    confidence = min(1.0, len(weekly_grades) / current_week)

                    prediction = {
                        'student_id': student_id,
                        'subject': subject,
                        'current_week': current_week,
                        'weeks_available': len(weekly_grades),
                        'current_avg_grade': float(current_avg),
                        'predicted_final_grade': float(final_grade_pred),
                        'prediction_confidence': float(confidence),
                        'trend': 'improving' if slope > 0.05 else 'declining' if slope < -0.05 else 'stable',
                        'trend_slope': float(slope)
                    }

                    # Добавляем группу если есть
                    if 'group' in subject_data.columns:
                        prediction['group'] = subject_data['group'].iloc[0]

                    predictions.append(prediction)

    if predictions:
        return pd.DataFrame(predictions).sort_values(['student_id', 'subject'])
    else:
        return pd.DataFrame()


def generate_recommendations(risk_factors: List[str]) -> List[str]:
    """
    Генерирует рекомендации на основе факторов риска.

    Parameters:
    -----------
    risk_factors : List[str]
        Список факторов риска

    Returns:
    --------
    List[str]
        Список рекомендаций
    """
    recommendations = []
    recommendations_map = {
        'Низкая средняя оценка': [
            "Обратиться за помощью к преподавателю",
            "Составить индивидуальный план обучения",
            "Увеличить время подготовки к занятиям"
        ],
        'Высокая изменчивость оценок': [
            "Проанализировать причины колебаний успеваемости",
            "Разработать стратегию стабильной подготовки",
            "Обратить внимание на организацию времени"
        ],
        'Снижение успеваемости': [
            "Провести диагностику причин снижения мотивации",
            "Установить краткосрочные учебные цели",
            "Увеличить частоту самопроверок"
        ],
        'Резкое падение успеваемости': [
            "Срочно обратиться к куратору группы",
            "Провести индивидуальную консультацию с преподавателем",
            "Пересмотреть учебную нагрузку"
        ],
        'Низкая посещаемость': [
            "Проанализировать причины пропусков",
            "Составить график посещения занятий",
            "Установить систему напоминаний"
        ]
    }

    for factor in risk_factors:
        for key, rec_list in recommendations_map.items():
            if key in factor:
                recommendations.extend(rec_list)

    # Удаляем дубликаты
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)

    # Ограничиваем 5 рекомендациями
    return unique_recommendations[:5]


def calculate_learning_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Вычисляет метрики обучения и эффективности образовательного процесса.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках

    Returns:
    --------
    Dict[str, Any]
        Метрики обучения
    """
    metrics = {}

    # 1. Общая эффективность
    total_students = df['student_id'].nunique()
    total_subjects = df['subject'].nunique()

    metrics['overall_efficiency'] = {
        'average_grade': float(df['grade'].mean()),
        'pass_rate': float((df['grade'] >= 5).mean() * 100),
        'excellence_rate': float((df['grade'] >= 9).mean() * 100),
        'failure_rate': float((df['grade'] < 5).mean() * 100),
        'student_subject_ratio': float(total_students / total_subjects) if total_subjects > 0 else 0
    }

    # 2. Распределение успеваемости
    grade_bins = pd.cut(df['grade'], bins=[0, 4, 6, 8, 10],
                        labels=['Неудовлетворительно', 'Удовлетворительно', 'Хорошо', 'Отлично'])
    grade_distribution = grade_bins.value_counts(normalize=True) * 100

    metrics['grade_distribution'] = grade_distribution.round(2).to_dict()

    # 3. Консистентность оценок
    if 'week' in df.columns:
        # Анализ стабильности оценок во времени
        weekly_variance = df.groupby('week')['grade'].std().mean()
        metrics['consistency'] = {
            'weekly_variance': float(weekly_variance),
            'stability_score': float(100 - min(100, weekly_variance * 10))  # 0-100 шкала
        }

    # 4. Эффективность по предметам
    subject_efficiency = {}
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        subject_metrics = {
            'average_grade': float(subject_data['grade'].mean()),
            'pass_rate': float((subject_data['grade'] >= 5).mean() * 100),
            'student_count': int(subject_data['student_id'].nunique()),
            'grade_std': float(subject_data['grade'].std())
        }
        subject_efficiency[subject] = subject_metrics

    metrics['subject_efficiency'] = subject_efficiency

    # 5. Метрики улучшения
    if 'week' in df.columns and df['week'].nunique() > 1:
        first_half = df[df['week'] <= df['week'].max() // 2]
        second_half = df[df['week'] > df['week'].max() // 2]

        if len(first_half) > 0 and len(second_half) > 0:
            improvement = second_half['grade'].mean() - first_half['grade'].mean()

            metrics['improvement_metrics'] = {
                'first_half_avg': float(first_half['grade'].mean()),
                'second_half_avg': float(second_half['grade'].mean()),
                'improvement': float(improvement),
                'improvement_percentage': float((improvement / first_half['grade'].mean()) * 100) if first_half[
                                                                                                         'grade'].mean() > 0 else 0
            }

    return metrics