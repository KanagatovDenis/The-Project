"""
Модуль для создания визуализаций образовательных данных
"""

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def create_grade_distribution(df: pd.DataFrame,
                              subject: Optional[str] = None,
                              group: Optional[str] = None,
                              bin_size: float = 1.0) -> go.Figure:
    """
    Создает гистограмму распределения оценок.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    subject : str, optional
        Фильтр по предмету
    group : str, optional
        Фильтр по группе
    bin_size : float
        Размер бина для гистограммы

    Returns:
    --------
    go.Figure
        График распределения оценок
    """
    # Фильтруем данные если указаны фильтры
    filtered_df = df.copy()

    if subject is not None:
        filtered_df = filtered_df[filtered_df['subject'] == subject]

    if group is not None and 'group' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['group'] == group]

    if len(filtered_df) == 0:
        raise ValueError("Нет данных для визуализации после фильтрации")

    # Создаем гистограмму
    fig = px.histogram(
        filtered_df,
        x='grade',
        nbins=int((filtered_df['grade'].max() - filtered_df['grade'].min()) / bin_size),
        title=f'Распределение оценок {f"по предмету {subject}" if subject else ""} {f"в группе {group}" if group else ""}',
        labels={'grade': 'Оценка', 'count': 'Количество'},
        color_discrete_sequence=['#6366F1'],
        opacity=0.8,
        marginal='box'
    )

    # Добавляем вертикальные линии для среднего и медианы
    mean_grade = filtered_df['grade'].mean()
    median_grade = filtered_df['grade'].median()

    fig.add_vline(
        x=mean_grade,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Среднее: {mean_grade:.2f}",
        annotation_position="top right"
    )

    fig.add_vline(
        x=median_grade,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Медиана: {median_grade:.2f}",
        annotation_position="bottom right"
    )

    # Настраиваем макет
    fig.update_layout(
        xaxis_title="Оценка",
        yaxis_title="Количество студентов",
        showlegend=False,
        bargap=0.1,
        plot_bgcolor='white',
        font=dict(size=12)
    )

    fig.update_xaxes(
        range=[0, 10.5],
        tickmode='linear',
        tick0=1,
        dtick=1
    )

    return fig


def create_performance_trend(df: pd.DataFrame,
                             student_ids: Optional[List[str]] = None,
                             subject: Optional[str] = None,
                             window: int = 3) -> go.Figure:
    """
    Создает график тренда успеваемости с скользящим средним.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    student_ids : List[str], optional
        Список ID студентов для отображения
    subject : str, optional
        Фильтр по предмету
    window : int
        Окно для скользящего среднего

    Returns:
    --------
    go.Figure
        График тренда успеваемости
    """
    # Подготавливаем данные
    trend_df = df.copy()

    if subject is not None:
        trend_df = trend_df[trend_df['subject'] == subject]

    if student_ids is not None:
        trend_df = trend_df[trend_df['student_id'].isin(student_ids)]

    if len(trend_df) == 0:
        raise ValueError("Нет данных для визуализации")

    # Группируем по неделе и студенту
    if 'week' not in trend_df.columns and 'date' in trend_df.columns:
        trend_df['week'] = trend_df['date'].dt.isocalendar().week

    if 'week' in trend_df.columns:
        # Создаем сводную таблицу
        pivot_data = trend_df.pivot_table(
            values='grade',
            index='week',
            columns='student_id',
            aggfunc='mean'
        ).reset_index()

        # Создаем график
        fig = go.Figure()

        # Добавляем линии для каждого студента
        for student_id in pivot_data.columns[1:]:  # Пропускаем колонку 'week'
            student_data = pivot_data[['week', student_id]].dropna()

            if len(student_data) > 0:
                # Добавляем скользящее среднее
                student_data['moving_avg'] = student_data[student_id].rolling(
                    window=min(window, len(student_data)),
                    min_periods=1
                ).mean()

                # Линия фактических оценок
                fig.add_trace(go.Scatter(
                    x=student_data['week'],
                    y=student_data[student_id],
                    mode='markers',
                    name=f'{student_id} (оценки)',
                    marker=dict(size=8, opacity=0.6),
                    showlegend=False
                ))

                # Линия скользящего среднего
                fig.add_trace(go.Scatter(
                    x=student_data['week'],
                    y=student_data['moving_avg'],
                    mode='lines',
                    name=student_id,
                    line=dict(width=3)
                ))

        # Настраиваем макет
        fig.update_layout(
            title=f'Динамика успеваемости {f"по предмету {subject}" if subject else ""}',
            xaxis_title='Неделя семестра',
            yaxis_title='Средняя оценка',
            hovermode='x unified',
            plot_bgcolor='white',
            height=500
        )

        fig.update_yaxes(range=[0, 10.5])

        return fig
    else:
        raise ValueError("Для создания тренда необходимы данные по неделям или датам")


def create_group_comparison(df: pd.DataFrame,
                            subjects: Optional[List[str]] = None) -> go.Figure:
    """
    Создает сравнительный анализ успеваемости между группами.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    subjects : List[str], optional
        Список предметов для сравнения

    Returns:
    --------
    go.Figure
        График сравнения групп
    """
    if 'group' not in df.columns:
        raise ValueError("Для сравнения групп необходима колонка 'group'")

    # Фильтруем данные если указаны предметы
    comparison_df = df.copy()

    if subjects is not None:
        comparison_df = comparison_df[comparison_df['subject'].isin(subjects)]

    # Группируем данные
    if subjects is None:
        # Сравниваем общую успеваемость по группам
        group_stats = comparison_df.groupby('group').agg({
            'grade': ['mean', 'std', 'count'],
            'student_id': 'nunique'
        }).round(2)

        group_stats.columns = ['mean_grade', 'std_grade', 'total_grades', 'unique_students']
        group_stats = group_stats.reset_index()

        # Создаем bar chart
        fig = px.bar(
            group_stats,
            x='group',
            y='mean_grade',
            error_y='std_grade',
            title='Сравнение успеваемости групп',
            labels={'mean_grade': 'Средняя оценка', 'group': 'Группа'},
            color='mean_grade',
            color_continuous_scale='Viridis',
            text='mean_grade'
        )

        # Добавляем информацию о количестве студентов
        for i, row in group_stats.iterrows():
            fig.add_annotation(
                x=row['group'],
                y=row['mean_grade'] + 0.2,
                text=f"{int(row['unique_students'])} студ.",
                showarrow=False,
                font=dict(size=10)
            )
    else:
        # Сравниваем по предметам
        subject_group_stats = comparison_df.groupby(['subject', 'group']).agg({
            'grade': 'mean'
        }).reset_index()

        # Создаем heatmap
        pivot_data = subject_group_stats.pivot(
            index='group',
            columns='subject',
            values='grade'
        )

        fig = px.imshow(
            pivot_data,
            title='Успеваемость по предметам и группам',
            labels=dict(x="Предмет", y="Группа", color="Средняя оценка"),
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )

        # Добавляем аннотации
        for i, group in enumerate(pivot_data.index):
            for j, subject in enumerate(pivot_data.columns):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{pivot_data.iloc[i, j]:.1f}",
                    showarrow=False,
                    font=dict(color='black' if pivot_data.iloc[i, j] > 5 else 'white')
                )

    fig.update_layout(
        plot_bgcolor='white',
        font=dict(size=12),
        coloraxis_showscale=True
    )

    return fig


def create_correlation_matrix(df: pd.DataFrame,
                              subjects: Optional[List[str]] = None) -> go.Figure:
    """
    Создает матрицу корреляции между предметами.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    subjects : List[str], optional
        Список предметов для анализа

    Returns:
    --------
    go.Figure
        Матрица корреляции
    """
    # Подготавливаем данные
    if subjects is None:
        subjects = df['subject'].unique()[:8]  # Ограничиваем 8 предметами

    # Создаем сводную таблицу: студент × предмет → средняя оценка
    pivot_df = df[df['subject'].isin(subjects)].pivot_table(
        index='student_id',
        columns='subject',
        values='grade',
        aggfunc='mean'
    )

    # Вычисляем корреляционную матрицу
    corr_matrix = pivot_df.corr().round(2)

    # Создаем heatmap
    fig = px.imshow(
        corr_matrix,
        title='Корреляция успеваемости по предметам',
        labels=dict(color="Коэффициент корреляции"),
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        aspect='auto'
    )

    # Добавляем аннотации
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                    size=10
                )
            )

    fig.update_layout(
        xaxis_title="Предмет",
        yaxis_title="Предмет",
        plot_bgcolor='white'
    )

    return fig


def create_risk_students_plot(df: pd.DataFrame,
                              threshold: float = 5.0,
                              min_records: int = 5) -> go.Figure:
    """
    Визуализирует студентов группы риска.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках
    threshold : float
        Порог для определения студентов группы риска
    min_records : int
        Минимальное количество записей для студента

    Returns:
    --------
    go.Figure
        График студентов группы риска
    """
    # Вычисляем статистику по студентам
    student_stats = df.groupby('student_id').agg({
        'grade': ['mean', 'count', 'std'],
        'subject': 'nunique'
    }).round(2)

    student_stats.columns = ['avg_grade', 'grade_count', 'grade_std', 'subject_count']
    student_stats = student_stats.reset_index()

    # Фильтруем студентов с достаточным количеством записей
    student_stats = student_stats[student_stats['grade_count'] >= min_records]

    # Определяем студентов группы риска
    student_stats['is_risk'] = student_stats['avg_grade'] < threshold

    # Создаем scatter plot
    fig = px.scatter(
        student_stats,
        x='grade_count',
        y='avg_grade',
        color='is_risk',
        size='subject_count',
        hover_name='student_id',
        hover_data=['grade_std', 'subject_count'],
        title='Анализ студентов группы риска',
        labels={
            'avg_grade': 'Средняя оценка',
            'grade_count': 'Количество оценок',
            'subject_count': 'Количество предметов',
            'is_risk': 'Группа риска'
        },
        color_discrete_map={True: '#EF4444', False: '#10B981'}
    )

    # Добавляем горизонтальную линию порога
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Порог риска: {threshold}",
        annotation_position="bottom right"
    )

    # Добавляем аннотации для студентов группы риска
    risk_students = student_stats[student_stats['is_risk']]

    for _, student in risk_students.iterrows():
        fig.add_annotation(
            x=student['grade_count'],
            y=student['avg_grade'],
            text=student['student_id'],
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=1,
            ax=0,
            ay=-40,
            font=dict(size=10, color='red')
        )

    fig.update_layout(
        plot_bgcolor='white',
        showlegend=True,
        height=600,
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    fig.update_xaxes(title_text="Количество оценок (активность)")
    fig.update_yaxes(title_text="Средняя оценка", range=[0, 10.5])

    return fig


def create_subject_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Создает комплексный анализ по предметам.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках

    Returns:
    --------
    go.Figure
        Комплексный график анализа предметов
    """
    # Вычисляем статистику по предметам
    subject_stats = df.groupby('subject').agg({
        'grade': ['mean', 'median', 'std', 'count'],
        'student_id': 'nunique'
    }).round(2)

    subject_stats.columns = ['mean', 'median', 'std', 'total_grades', 'unique_students']
    subject_stats = subject_stats.reset_index()

    # Сортируем по средней оценке
    subject_stats = subject_stats.sort_values('mean', ascending=True)

    # Создаем subplot с 2 рядами
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Средняя оценка по предметам',
            'Разброс оценок',
            'Количество оценок',
            'Распределение по предметам'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # 1. Bar chart: средние оценки
    fig.add_trace(
        go.Bar(
            x=subject_stats['mean'],
            y=subject_stats['subject'],
            orientation='h',
            marker_color='#6366F1',
            name='Средняя оценка',
            text=subject_stats['mean'],
            textposition='auto'
        ),
        row=1, col=1
    )

    # 2. Scatter: среднее vs стандартное отклонение
    fig.add_trace(
        go.Scatter(
            x=subject_stats['mean'],
            y=subject_stats['std'],
            mode='markers+text',
            marker=dict(
                size=subject_stats['unique_students'] / subject_stats['unique_students'].max() * 30 + 10,
                color=subject_stats['mean'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Средняя оценка")
            ),
            text=subject_stats['subject'],
            textposition='top center',
            name='Разброс оценок'
        ),
        row=1, col=2
    )

    # 3. Bar chart: количество оценок
    fig.add_trace(
        go.Bar(
            x=subject_stats['subject'],
            y=subject_stats['total_grades'],
            marker_color='#10B981',
            name='Количество оценок'
        ),
        row=2, col=1
    )

    # 4. Box plot: распределение по предметам
    for subject in subject_stats['subject'][:6]:  # Ограничиваем 6 предметами
        subject_data = df[df['subject'] == subject]['grade']
        fig.add_trace(
            go.Box(
                y=subject_data,
                name=subject[:15],  # Обрезаем длинные названия
                boxpoints='outliers',
                jitter=0.3
            ),
            row=2, col=2
        )

    # Настраиваем макет
    fig.update_layout(
        title_text='Комплексный анализ предметов',
        showlegend=False,
        height=800,
        plot_bgcolor='white'
    )

    # Настраиваем оси
    fig.update_xaxes(title_text="Средняя оценка", row=1, col=1)
    fig.update_xaxes(title_text="Средняя оценка", row=1, col=2)
    fig.update_xaxes(title_text="Предмет", row=2, col=1)
    fig.update_xaxes(title_text="Предмет", row=2, col=2)

    fig.update_yaxes(title_text="Предмет", row=1, col=1)
    fig.update_yaxes(title_text="Стандартное отклонение", row=1, col=2)
    fig.update_yaxes(title_text="Количество оценок", row=2, col=1)
    fig.update_yaxes(title_text="Оценка", range=[0, 10.5], row=2, col=2)

    return fig


def create_student_portfolio(student_id: str, df: pd.DataFrame) -> go.Figure:
    """
    Создает портфолио успеваемости конкретного студента.

    Parameters:
    -----------
    student_id : str
        ID студента
    df : pd.DataFrame
        DataFrame с данными об оценках

    Returns:
    --------
    go.Figure
        Портфолио студента
    """
    # Фильтруем данные студента
    student_data = df[df['student_id'] == student_id].copy()

    if len(student_data) == 0:
        raise ValueError(f"Студент {student_id} не найден")

    # Вычисляем статистику
    avg_grade = student_data['grade'].mean()
    total_subjects = student_data['subject'].nunique()
    total_grades = len(student_data)

    # Создаем subplot
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Успеваемость по предметам (среднее: {avg_grade:.2f})',
            'Динамика успеваемости',
            'Распределение оценок',
            'Сравнение с группой'
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "box"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # 1. Bar chart: оценки по предметам
    subject_grades = student_data.groupby('subject')['grade'].mean().sort_values()

    fig.add_trace(
        go.Bar(
            x=subject_grades.values,
            y=subject_grades.index,
            orientation='h',
            marker_color='#6366F1',
            name='Средняя оценка',
            text=[f'{v:.1f}' for v in subject_grades.values],
            textposition='auto'
        ),
        row=1, col=1
    )

    # 2. Line chart: динамика успеваемости
    if 'week' in student_data.columns or 'date' in student_data.columns:
        if 'week' not in student_data.columns:
            student_data['week'] = student_data['date'].dt.isocalendar().week

        weekly_grades = student_data.groupby('week')['grade'].mean().reset_index()

        fig.add_trace(
            go.Scatter(
                x=weekly_grades['week'],
                y=weekly_grades['grade'],
                mode='lines+markers',
                line=dict(width=3, color='#10B981'),
                marker=dict(size=10),
                name='Средняя оценка за неделю'
            ),
            row=1, col=2
        )

    # 3. Histogram: распределение оценок
    fig.add_trace(
        go.Histogram(
            x=student_data['grade'],
            nbinsx=10,
            marker_color='#8B5CF6',
            name='Распределение оценок'
        ),
        row=2, col=1
    )

    # 4. Box plot: сравнение с группой
    if 'group' in student_data.columns:
        group = student_data['group'].iloc[0]
        group_data = df[df['group'] == group]

        # Данные студента
        fig.add_trace(
            go.Box(
                y=student_data['grade'],
                name='Студент',
                marker_color='#EF4444',
                boxpoints='all'
            ),
            row=2, col=2
        )

        # Данные группы
        fig.add_trace(
            go.Box(
                y=group_data['grade'],
                name='Группа',
                marker_color='#3B82F6'
            ),
            row=2, col=2
        )

    # Настраиваем макет
    fig.update_layout(
        title_text=f'Портфолио успеваемости: {student_id}',
        showlegend=True,
        height=700,
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    # Обновляем подписи осей
    fig.update_xaxes(title_text="Средняя оценка", row=1, col=1, range=[0, 10.5])
    fig.update_xaxes(title_text="Неделя", row=1, col=2)
    fig.update_xaxes(title_text="Оценка", row=2, col=1, range=[0, 10.5])
    fig.update_xaxes(title_text="", row=2, col=2)

    fig.update_yaxes(title_text="Предмет", row=1, col=1)
    fig.update_yaxes(title_text="Оценка", row=1, col=2, range=[0, 10.5])
    fig.update_yaxes(title_text="Количество", row=2, col=1)
    fig.update_yaxes(title_text="Оценка", row=2, col=2, range=[0, 10.5])

    return fig


def create_interactive_dashboard(df: pd.DataFrame) -> go.Figure:
    """
    Создает интерактивную панель управления с несколькими визуализациями.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными об оценках

    Returns:
    --------
    go.Figure
        Интерактивная панель управления
    """
    # Создаем комплексную фигуру с 6 визуализациями
    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Распределение оценок',
            'Динамика успеваемости',
            'Сравнение групп',
            'Студенты группы риска',
            'Корреляция предметов',
            'Топ-5 студентов'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ]
    )

    # 1. Распределение оценок (гистограмма)
    grade_dist = px.histogram(
        df, x='grade', nbins=10,
        color_discrete_sequence=['#6366F1']
    )
    fig.add_trace(grade_dist.data[0], row=1, col=1)

    # 2. Динамика успеваемости (линейный график)
    if 'week' in df.columns:
        weekly_avg = df.groupby('week')['grade'].mean().reset_index()
        trend_line = px.line(weekly_avg, x='week', y='grade')
        fig.add_trace(trend_line.data[0], row=1, col=2)

    # 3. Сравнение групп (столбчатая диаграмма)
    if 'group' in df.columns:
        group_stats = df.groupby('group')['grade'].mean().reset_index()
        group_bars = px.bar(
            group_stats, x='group', y='grade',
            color='grade', color_continuous_scale='Viridis'
        )
        fig.add_trace(group_bars.data[0], row=2, col=1)

    # 4. Студенты группы риска (точечная диаграмма)
    student_stats = df.groupby('student_id').agg({
        'grade': ['mean', 'count']
    }).round(2)
    student_stats.columns = ['avg_grade', 'grade_count']
    student_stats = student_stats.reset_index()

    student_stats['is_risk'] = student_stats['avg_grade'] < 5.0

    risk_scatter = px.scatter(
        student_stats,
        x='grade_count',
        y='avg_grade',
        color='is_risk',
        hover_name='student_id',
        color_discrete_map={True: '#EF4444', False: '#10B981'}
    )
    fig.add_trace(risk_scatter.data[0], row=2, col=2)
    if len(risk_scatter.data) > 1:
        fig.add_trace(risk_scatter.data[1], row=2, col=2)

    # 5. Корреляция предметов (тепловая карта)
    try:
        subjects = df['subject'].unique()[:6]
        pivot_df = df[df['subject'].isin(subjects)].pivot_table(
            index='student_id',
            columns='subject',
            values='grade',
            aggfunc='mean'
        )
        corr_matrix = pivot_df.corr().round(2)

        correlation_heatmap = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        fig.add_trace(correlation_heatmap.data[0], row=3, col=1)
    except:
        # Если не удалось создать тепловую карту, добавляем placeholder
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=['Недостаточно данных']), row=3, col=1)

    # 6. Топ-5 студентов (столбчатая диаграмма)
    top_students = student_stats.nlargest(5, 'avg_grade')
    top_bars = px.bar(
        top_students,
        x='student_id',
        y='avg_grade',
        color='avg_grade',
        color_continuous_scale='Greens'
    )
    fig.add_trace(top_bars.data[0], row=3, col=2)

    # Настраиваем макет
    fig.update_layout(
        title_text='📊 Панель управления успеваемостью',
        showlegend=False,
        height=1000,
        plot_bgcolor='white',
        font=dict(size=11)
    )

    # Настраиваем оси
    fig.update_xaxes(title_text="Оценка", row=1, col=1)
    fig.update_xaxes(title_text="Неделя", row=1, col=2)
    fig.update_xaxes(title_text="Группа", row=2, col=1)
    fig.update_xaxes(title_text="Количество оценок", row=2, col=2)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_xaxes(title_text="Студент", row=3, col=2)

    fig.update_yaxes(title_text="Количество", row=1, col=1)
    fig.update_yaxes(title_text="Средняя оценка", row=1, col=2, range=[0, 10.5])
    fig.update_yaxes(title_text="Средняя оценка", row=2, col=1, range=[0, 10.5])
    fig.update_yaxes(title_text="Средняя оценка", row=2, col=2, range=[0, 10.5])
    fig.update_yaxes(title_text="", row=3, col=1)
    fig.update_yaxes(title_text="Средняя оценка", row=3, col=2, range=[0, 10.5])

    return fig


def save_visualization(fig: go.Figure, filepath: str,
                       width: int = 1200, height: int = 600) -> None:
    """
    Сохраняет визуализацию в файл.

    Parameters:
    -----------
    fig : go.Figure
        Объект Figure для сохранения
    filepath : str
        Путь для сохранения файла
    width : int
        Ширина изображения
    height : int
        Высота изображения
    """
    import plotly.io as pio

    # Определяем формат по расширению файла
    filepath = str(filepath)

    if filepath.endswith('.html'):
        fig.write_html(filepath)
    elif filepath.endswith('.png'):
        fig.write_image(filepath, width=width, height=height)
    elif filepath.endswith('.pdf'):
        fig.write_image(filepath, width=width, height=height)
    elif filepath.endswith('.svg'):
        fig.write_image(filepath, width=width, height=height)
    else:
        # По умолчанию сохраняем как HTML
        fig.write_html(filepath if filepath.endswith('.html') else filepath + '.html')

    print(f"✅ Визуализация сохранена в {filepath}")