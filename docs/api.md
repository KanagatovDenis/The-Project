# API документация EduViz Dashboard

## Обзор

EduViz Dashboard предоставляет Python API для работы с образовательными данными. Все модули доступны через пакет `src`.

## Модуль `data_loader`

### Основные функции:

#### `load_student_data(filepath: str, **kwargs) -> pd.DataFrame`
Загружает данные об успеваемости из файла.

**Параметры:**
- `filepath`: Путь к файлу с данными
- `**kwargs`: Дополнительные параметры для pd.read_csv

**Возвращает:**
- DataFrame с данными об успеваемости

**Пример:**
```python
from src.data_loader import load_student_data

df = load_student_data('data/raw/grades.csv', encoding='utf-8')