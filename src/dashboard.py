"""
Модуль для создания интерактивного Dash-дашборда
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Импортируем наши модули
from .data_loader import load_student_data, clean_data
from .visualizer import (
    create_grade_distribution,
    create_performance_trend,
    create_group_comparison,
    create_correlation_matrix,
    create_risk_students_plot,
    create_subject_analysis,
    create_student_portfolio
)
from .analyzer import analyze_performance, identify_at_risk_students

# Инициализируем Dash приложение
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'],
    suppress_callback_exceptions=True,
    title='EduViz Dashboard'
)

server = app.server


def create_dashboard(df: pd.DataFrame = None):
    """
    Создает и настраивает Dash дашборд.

    Parameters:
    -----------
    df : pd.DataFrame, optional
        DataFrame с данными об оценках

    Returns:
    --------
    dash.Dash
        Настроенное Dash приложение
    """
    # Если данные не предоставлены, создаем пустой DataFrame
    if df is None:
        df = pd.DataFrame()

    # Навигационная панель
    navbar = dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("📊 EduViz Dashboard", className="ms-2 text-white"),
                    html.P("Визуализация образовательных данных", className="mb-0 text-light")
                ], width="auto"),
            ], align="center"),

            dbc.NavbarToggler(id="navbar-toggler"),

            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("📈 Обзор", href="#", id="tab-overview")),
                    dbc.NavItem(dbc.NavLink("🎓 Студенты", href="#", id="tab-students")),
                    dbc.NavItem(dbc.NavLink("📚 Предметы", href="#", id="tab-subjects")),
                    dbc.NavItem(dbc.NavLink("⚠️  Риски", href="#", id="tab-risks")),
                    dbc.NavItem(dbc.NavLink("⚙️  Настройки", href="#", id="tab-settings")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),

            html.Div([
                dbc.Badge(f"Студентов: {df['student_id'].nunique() if not df.empty else 0}",
                         color="light", className="me-2"),
                dbc.Badge(f"Оценок: {len(df) if not df.empty else 0}",
                         color="light", className="me-2"),
                dbc.Badge(f"Предметов: {df['subject'].nunique() if not df.empty else 0}",
                         color="light"),
            ], className="d-flex align-items-center"),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4 shadow"
    )

    # Боковая панель с фильтрами
    sidebar = dbc.Card([
        dbc.CardHeader("🔍 Фильтры данных", className="fw-bold"),
        dbc.CardBody([
            html.Div([
                html.Label("Предмет:", className="form-label"),
                dcc.Dropdown(
                    id='subject-filter',
                    options=[{'label': 'Все предметы', 'value': 'all'}] +
                            [{'label': subj, 'value': subj} for subj in sorted(df['subject'].unique())] if not df.empty else [],
                    value='all',
                    clearable=False,
                    className="mb-3"
                ),

                html.Label("Группа:", className="form-label"),
                dcc.Dropdown(
                    id='group-filter',
                    options=[{'label': 'Все группы', 'value': 'all'}] +
                            [{'label': grp, 'value': grp} for grp in sorted(df['group'].unique())] if 'group' in df.columns and not df.empty else [],
                    value='all',
                    clearable=False,
                    className="mb-3"
                ),

                html.Label("Диапазон оценок:", className="form-label"),
                dcc.RangeSlider(
                    id='grade-range',
                    min=1,
                    max=10,
                    step=0.5,
                    marks={i: str(i) for i in range(1, 11)},
                    value=[1, 10],
                    className="mb-3"
                ),

                html.Label("Период (недели):", className="form-label"),
                dcc.RangeSlider(
                    id='week-range',
                    min=df['week'].min() if 'week' in df.columns and not df.empty else 1,
                    max=df['week'].max() if 'week' in df.columns and not df.empty else 16,
                    step=1,
                    marks={i: str(i) for i in range(1, 17, 2)} if not df.empty else {},
                    value=[df['week'].min() if 'week' in df.columns and not df.empty else 1,
                          df['week'].max() if 'week' in df.columns and not df.empty else 16],
                    className="mb-3",
                    disabled='week' not in df.columns or df.empty
                ),

                dbc.Button("Применить фильтры",
                          id='apply-filters',
                          color="primary",
                          className="w-100 mt-3",
                          n_clicks=0)
            ])
        ]),

        dbc.CardFooter([
            html.Small("Используйте фильтры для настройки отображаемых данных",
                      className="text-muted"),
            html.Br(),
            dbc.Button("Сбросить фильтры",
                      id='reset-filters',
                      color="outline-secondary",
                      size="sm",
                      className="w-100 mt-2")
        ])
    ], className="shadow-sm")

    # Карточки с ключевыми метриками
    if not df.empty:
        analysis = analyze_performance(df)
        overall_stats = analysis['overall']

        metrics_cards = dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{overall_stats['mean_grade']:.2f}", className="card-title"),
                        html.P("Средняя оценка", className="card-text text-muted"),
                        html.Small(f"Медиана: {overall_stats['median_grade']:.2f}",
                                 className="text-success" if overall_stats['mean_grade'] >= 6 else "text-warning")
                    ])
                ], className="text-center shadow-sm border-success"),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{overall_stats['total_students']}", className="card-title"),
                        html.P("Студентов", className="card-text text-muted"),
                        html.Small(f"Групп: {overall_stats.get('total_groups', 'N/A')}",
                                 className="text-info")
                    ])
                ], className="text-center shadow-sm border-info"),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{overall_stats['total_subjects']}", className="card-title"),
                        html.P("Предметов", className="card-text text-muted"),
                        html.Small("Активно", className="text-primary")
                    ])
                ], className="text-center shadow-sm border-primary"),
                width=3
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(analysis['risk_students'])}", className="card-title"),
                        html.P("Студентов в группе риска", className="card-text text-muted"),
                        html.Small(f"{len(analysis['risk_students'])/overall_stats['total_students']*100:.1f}%",
                                 className="text-danger")
                    ])
                ], className="text-center shadow-sm border-danger"),
                width=3
            ),
        ], className="mb-4")
    else:
        metrics_cards = dbc.Alert("Данные не загружены. Используйте генератор тестовых данных.",
                                 color="warning")

    # Основное содержимое с вкладками
    content_tabs = dbc.Tabs([
        dbc.Tab([
            html.Div([
                metrics_cards,

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📊 Распределение оценок"),
                            dbc.CardBody([
                                dcc.Graph(id='grade-distribution-chart')
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📈 Динамика успеваемости"),
                            dbc.CardBody([
                                dcc.Graph(id='performance-trend-chart')
                            ])
                        ], className="shadow-sm mb-4")
                    ], width=6),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🏫 Сравнение групп"),
                            dbc.CardBody([
                                dcc.Graph(id='group-comparison-chart')
                            ])
                        ], className="shadow-sm")
                    ], width=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🔗 Корреляция предметов"),
                            dbc.CardBody([
                                dcc.Graph(id='correlation-chart')
                            ])
                        ], className="shadow-sm")
                    ], width=6),
                ]),
            ])
        ], label="Обзор", tab_id="overview"),

        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("🎯 Анализ студента"),
                            dbc.CardBody([
                                html.Label("Выберите студента:", className="form-label"),
                                dcc.Dropdown(
                                    id='student-selector',
                                    options=[{'label': sid, 'value': sid}
                                            for sid in sorted(df['student_id'].unique())] if not df.empty else [],
                                    value=df['student_id'].iloc[0] if not df.empty else None,
                                    className="mb-3"
                                ),
                                dcc.Graph(id='student-portfolio-chart')
                            ])
                        ], className="shadow-sm")
                    ], width=12),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📋 Топ студентов"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id='top-students-table',
                                    columns=[
                                        {'name': 'Студент', 'id': 'student_id'},
                                        {'name': 'Средняя оценка', 'id': 'avg_grade'},
                                        {'name': 'Количество оценок', 'id': 'grade_count'},
                                        {'name': 'Предметы', 'id': 'subject_count'},
                                        {'name': 'Группа', 'id': 'group'}
                                    ] if 'group' in df.columns else [
                                        {'name': 'Студент', 'id': 'student_id'},
                                        {'name': 'Средняя оценка', 'id': 'avg_grade'},
                                        {'name': 'Количество оценок', 'id': 'grade_count'},
                                        {'name': 'Предметы', 'id': 'subject_count'}
                                    ],
                                    style_table={'overflowX': 'auto'},
                                    style_cell={'textAlign': 'center'},
                                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                    page_size=10
                                )
                            ])
                        ], className="shadow-sm mt-4")
                    ], width=12),
                ]),
            ])
        ], label="Студенты", tab_id="students"),

        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📚 Анализ по предметам"),
                            dbc.CardBody([
                                dcc.Graph(id='subject-analysis-chart')
                            ])
                        ], className="shadow-sm")
                    ], width=12),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📊 Статистика по предметам"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id='subject-stats-table',
                                    columns=[
                                        {'name': 'Предмет', 'id': 'subject'},
                                        {'name': 'Средняя оценка', 'id': 'mean_grade'},
                                        {'name': 'Медиана', 'id': 'median_grade'},
                                        {'name': 'Станд. отклонение', 'id': 'std_grade'},
                                        {'name': 'Студентов', 'id': 'student_count'},
                                        {'name': '% Успеваемость', 'id': 'pass_rate'}
                                    ],
                                    style_table={'overflowX': 'auto'},
                                    style_cell={'textAlign': 'center'},
                                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                    page_size=8
                                )
                            ])
                        ], className="shadow-sm mt-4")
                    ], width=12),
                ]),
            ])
        ], label="Предметы", tab_id="subjects"),

        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("⚠️  Студенты группы риска"),
                            dbc.CardBody([
                                dcc.Graph(id='risk-students-chart'),
                                html.Div(id='risk-students-details', className="mt-3")
                            ])
                        ], className="shadow-sm")
                    ], width=12),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📋 Детальный анализ рисков"),
                            dbc.CardBody([
                                dash_table.DataTable(
                                    id='risk-students-table',
                                    columns=[
                                        {'name': 'Студент', 'id': 'student_id'},
                                        {'name': 'Средняя оценка', 'id': 'avg_grade'},
                                        {'name': 'Факторы риска', 'id': 'risk_factors_count'},
                                        {'name': 'Уровень риска', 'id': 'risk_level'},
                                        {'name': 'Рекомендации', 'id': 'recommendations'}
                                    ],
                                    style_table={'overflowX': 'auto'},
                                    style_cell={
                                        'textAlign': 'left',
                                        'whiteSpace': 'normal',
                                        'height': 'auto',
                                        'minWidth': '100px'
                                    },
                                    style_header={'backgroundColor': 'rgb(255, 230, 230)', 'fontWeight': 'bold'},
                                    page_size=10,
                                    filter_action="native",
                                    sort_action="native"
                                )
                            ])
                        ], className="shadow-sm mt-4")
                    ], width=12),
                ]),
            ])
        ], label="Риски", tab_id="risks"),

        dbc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("⚙️  Настройки дашборда"),
                            dbc.CardBody([
                                html.H5("Параметры визуализации", className="mb-3"),

                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Цветовая схема:", className="form-label"),
                                        dcc.Dropdown(
                                            id='color-scheme',
                                            options=[
                                                {'label': 'Plotly', 'value': 'plotly'},
                                                {'label': 'Viridis', 'value': 'viridis'},
                                                {'label': 'Plasma', 'value': 'plasma'},
                                                {'label': 'Теплая', 'value': 'warm'},
                                                {'label': 'Холодная', 'value': 'cool'}
                                            ],
                                            value='plotly',
                                            className="mb-3"
                                        ),
                                    ], width=6),

                                    dbc.Col([
                                        html.Label("Размер шрифта:", className="form-label"),
                                        dcc.Slider(
                                            id='font-size',
                                            min=10,
                                            max=20,
                                            step=1,
                                            value=14,
                                            marks={i: str(i) for i in range(10, 21, 2)},
                                            className="mb-3"
                                        ),
                                    ], width=6),
                                ]),

                                html.H5("Экспорт данных", className="mt-4 mb-3"),

                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "📥 Экспорт в CSV",
                                            id='export-csv',
                                            color="success",
                                            className="w-100 mb-2"
                                        ),
                                    ], width=4),

                                    dbc.Col([
                                        dbc.Button(
                                            "📊 Экспорт графиков",
                                            id='export-charts',
                                            color="info",
                                            className="w-100 mb-2"
                                        ),
                                    ], width=4),

                                    dbc.Col([
                                        dbc.Button(
                                            "📄 Генерация отчета",
                                            id='generate-report',
                                            color="primary",
                                            className="w-100 mb-2"
                                        ),
                                    ], width=4),
                                ]),

                                html.Div(id='export-status', className="mt-3"),
                            ])
                        ], className="shadow-sm")
                    ], width=12),
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("ℹ️  Информация о системе"),
                            dbc.CardBody([
                                html.P(f"Версия дашборда: 1.0.0"),
                                html.P(f"Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                                html.P(f"Загружено записей: {len(df) if not df.empty else 0}"),
                                html.P(f"Размер данных: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB" if not df.empty else "Данные не загружены"),
                                html.Hr(),
                                html.Small("EduViz Dashboard © 2024 - Система визуализации образовательных данных",
                                         className="text-muted")
                            ])
                        ], className="shadow-sm mt-4")
                    ], width=12),
                ]),
            ])
        ], label="Настройки", tab_id="settings"),
    ], id="content-tabs", active_tab="overview", className="mt-3")

    # Основной layout
    app.layout = dbc.Container([
        navbar,

        dbc.Row([
            dbc.Col(sidebar, width=3, className="mb-4"),
            dbc.Col(content_tabs, width=9),
        ]),

        # Скрытые элементы для хранения данных
        dcc.Store(id='filtered-data'),
        dcc.Store(id='original-data', data=df.to_dict('records') if not df.empty else {}),
        dcc.Store(id='analysis-results'),

        # Модальное окно для сообщений
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Уведомление")),
            dbc.ModalBody(id="modal-body"),
            dbc.ModalFooter(
                dbc.Button("Закрыть", id="close-modal", className="ms-auto", n_clicks=0)
            ),
        ], id="notification-modal", is_open=False),
    ], fluid=True, className="p-3")

    return app


# Callbacks для интерактивности
@app.callback(
    Output('filtered-data', 'data'),
    [Input('apply-filters', 'n_clicks'),
     Input('reset-filters', 'n_clicks')],
    [State('original-data', 'data'),
     State('subject-filter', 'value'),
     State('group-filter', 'value'),
     State('grade-range', 'value'),
     State('week-range', 'value')]
)
def update_filtered_data(apply_clicks, reset_clicks, original_data, subject, group, grade_range, week_range):
    """
    Обновляет отфильтрованные данные на основе выбранных фильтров.
    """
    ctx = dash.callback_context

    if not original_data or len(original_data) == 0:
        return {}

    df = pd.DataFrame(original_data)

    # Если нажата кнопка сброса
    if ctx.triggered_id == 'reset-filters':
        return df.to_dict('records')

    # Применяем фильтры
    filtered_df = df.copy()

    # Фильтр по предмету
    if subject != 'all':
        filtered_df = filtered_df[filtered_df['subject'] == subject]

    # Фильтр по группе
    if group != 'all' and 'group' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['group'] == group]

    # Фильтр по диапазону оценок
    filtered_df = filtered_df[
        (filtered_df['grade'] >= grade_range[0]) &
        (filtered_df['grade'] <= grade_range[1])
    ]

    # Фильтр по неделям
    if 'week' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['week'] >= week_range[0]) &
            (filtered_df['week'] <= week_range[1])
        ]

    return filtered_df.to_dict('records')


@app.callback(
    Output('grade-distribution-chart', 'figure'),
    [Input('filtered-data', 'data'),
     Input('subject-filter', 'value')]
)
def update_grade_distribution(filtered_data, subject_filter):
    """
    Обновляет график распределения оценок.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure()

    df = pd.DataFrame(filtered_data)

    # Определяем предмет для графика
    subject = None if subject_filter == 'all' else subject_filter

    try:
        fig = create_grade_distribution(df, subject=subject)
        return fig
    except Exception as e:
        return go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})


@app.callback(
    Output('performance-trend-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_performance_trend(filtered_data):
    """
    Обновляет график тренда успеваемости.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure()

    df = pd.DataFrame(filtered_data)

    try:
        # Выбираем топ-5 студентов для отображения
        student_stats = df.groupby('student_id').agg({
            'grade': 'mean',
            'student_id': 'count'
        })
        student_stats.columns = ['avg_grade', 'count']
        top_students = student_stats.nlargest(5, 'avg_grade').index.tolist()

        fig = create_performance_trend(df, student_ids=top_students)
        return fig
    except Exception as e:
        return go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})


@app.callback(
    Output('group-comparison-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_group_comparison(filtered_data):
    """
    Обновляет график сравнения групп.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure()

    df = pd.DataFrame(filtered_data)

    if 'group' not in df.columns:
        return go.Figure(data=[], layout={'title': 'Данные о группах отсутствуют'})

    try:
        fig = create_group_comparison(df)
        return fig
    except Exception as e:
        return go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})


@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_correlation_matrix(filtered_data):
    """
    Обновляет матрицу корреляции.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure()

    df = pd.DataFrame(filtered_data)

    try:
        # Ограничиваем количество предметов для лучшей читаемости
        subjects = df['subject'].unique()[:6]
        fig = create_correlation_matrix(df, subjects=subjects)
        return fig
    except Exception as e:
        return go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})


@app.callback(
    [Output('student-portfolio-chart', 'figure'),
     Output('top-students-table', 'data')],
    [Input('filtered-data', 'data'),
     Input('student-selector', 'value')]
)
def update_student_info(filtered_data, selected_student):
    """
    Обновляет портфолио студента и таблицу топ-студентов.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure(), []

    df = pd.DataFrame(filtered_data)

    # Обновляем портфолио студента
    portfolio_fig = go.Figure()
    if selected_student:
        try:
            portfolio_fig = create_student_portfolio(selected_student, df)
        except Exception as e:
            portfolio_fig = go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})

    # Обновляем таблицу топ-студентов
    student_stats = df.groupby('student_id').agg({
        'grade': ['mean', 'count'],
        'subject': 'nunique'
    }).round(2)

    student_stats.columns = ['avg_grade', 'grade_count', 'subject_count']
    student_stats = student_stats.reset_index()

    # Добавляем информацию о группе если есть
    if 'group' in df.columns:
        group_info = df.groupby('student_id')['group'].first()
        student_stats = student_stats.merge(group_info, on='student_id', how='left')

    # Сортируем и берем топ-10
    top_students = student_stats.nlargest(10, 'avg_grade')

    return portfolio_fig, top_students.to_dict('records')


@app.callback(
    [Output('subject-analysis-chart', 'figure'),
     Output('subject-stats-table', 'data')],
    [Input('filtered-data', 'data')]
)
def update_subject_analysis(filtered_data):
    """
    Обновляет анализ по предметам.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure(), []

    df = pd.DataFrame(filtered_data)

    # Создаем график анализа предметов
    try:
        analysis_fig = create_subject_analysis(df)
    except Exception as e:
        analysis_fig = go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})

    # Создаем таблицу статистики по предметам
    subject_stats = []
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        stats = {
            'subject': subject,
            'mean_grade': round(subject_data['grade'].mean(), 2),
            'median_grade': round(subject_data['grade'].median(), 2),
            'std_grade': round(subject_data['grade'].std(), 2),
            'student_count': subject_data['student_id'].nunique(),
            'pass_rate': round((subject_data['grade'] >= 5).mean() * 100, 1)
        }
        subject_stats.append(stats)

    # Сортируем по средней оценке
    subject_stats.sort(key=lambda x: x['mean_grade'], reverse=True)

    return analysis_fig, subject_stats


@app.callback(
    [Output('risk-students-chart', 'figure'),
     Output('risk-students-table', 'data')],
    [Input('filtered-data', 'data')]
)
def update_risk_analysis(filtered_data):
    """
    Обновляет анализ студентов группы риска.
    """
    if not filtered_data or len(filtered_data) == 0:
        return go.Figure(), []

    df = pd.DataFrame(filtered_data)

    # Создаем график студентов группы риска
    try:
        risk_fig = create_risk_students_plot(df)
    except Exception as e:
        risk_fig = go.Figure(data=[], layout={'title': f'Ошибка: {str(e)}'})

    # Идентифицируем студентов группы риска
    try:
        risk_df = identify_at_risk_students(df)

        if not risk_df.empty:
            # Форматируем данные для таблицы
            risk_data = []
            for _, row in risk_df.iterrows():
                risk_entry = {
                    'student_id': row['student_id'],
                    'avg_grade': round(row['avg_grade'], 2),
                    'risk_factors_count': len(row['risk_factors']),
                    'risk_level': 'Высокий' if row['risk_score'] >= 3 else 'Средний' if row['risk_score'] == 2 else 'Низкий',
                    'recommendations': ', '.join(row['recommendations'][:2])  # Первые 2 рекомендации
                }
                risk_data.append(risk_entry)
        else:
            risk_data = []
    except Exception as e:
        risk_data = []

    return risk_fig, risk_data


@app.callback(
    [Output('notification-modal', 'is_open'),
     Output('modal-body', 'children')],
    [Input('export-csv', 'n_clicks'),
     Input('export-charts', 'n_clicks'),
     Input('generate-report', 'n_clicks'),
     Input('close-modal', 'n_clicks')],
    [State('filtered-data', 'data'),
     State('notification-modal', 'is_open')]
)
def handle_export_actions(csv_clicks, charts_clicks, report_clicks, close_clicks, filtered_data, is_open):
    """
    Обрабатывает действия экспорта и генерации отчетов.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open, ""

    button_id = ctx.triggered_id

    if button_id == 'close-modal':
        return False, ""

    if not filtered_data or len(filtered_data) == 0:
        return True, "❌ Нет данных для экспорта"

    df = pd.DataFrame(filtered_data)

    if button_id == 'export-csv':
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'eduviz_export_{timestamp}.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            return True, f"✅ Данные экспортированы в файл: {filename}"
        except Exception as e:
            return True, f"❌ Ошибка экспорта: {str(e)}"

    elif button_id == 'export-charts':
        return True, "📊 Экспорт графиков в разработке..."

    elif button_id == 'generate-report':
        return True, "📄 Генерация отчета в разработке..."

    return is_open, ""


# Callback для обновления доступных студентов в селекторе
@app.callback(
    Output('student-selector', 'options'),
    [Input('filtered-data', 'data')]
)
def update_student_options(filtered_data):
    """
    Обновляет список доступных студентов в селекторе.
    """
    if not filtered_data or len(filtered_data) == 0:
        return []

    df = pd.DataFrame(filtered_data)
    students = sorted(df['student_id'].unique())
    return [{'label': sid, 'value': sid} for sid in students]


# Callback для обновления статистики в навигационной панели
@app.callback(
    [Output('navbar-collapse', 'is_open')],
    [Input('navbar-toggler', 'n_clicks')],
    [State('navbar-collapse', 'is_open')]
)
def toggle_navbar_collapse(n_clicks, is_open):
    """
    Переключает состояние collapse навигационной панели.
    """
    if n_clicks:
        return [not is_open]
    return [is_open]