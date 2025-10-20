import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer



# --- Очистка датасета: только удаление строк с пропусками ---
def clean_dataset(df):
    return df.dropna().copy()

def remove_outliers_iqr(df: pd.DataFrame, numeric_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Удаляет выбросы по IQR для числовых признаков.
    factor=1.5 соответствует правилу Тьюки.
    """
    df_clean = df.copy()
    for col in numeric_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


# --- Парсинг ST_PROD ---

def add_standard_features_v2(df: pd.DataFrame, col: str = "ST_PROD") -> pd.DataFrame:
    """
    Разбирает столбец с описанием стандарта на:
      - system: система стандарта (ГОСТ, EN, ASTM, СТО, ISO, DIN, ТУ, ...)
      - code: код без года и пометок изменений (например, '10025-3', '27772', '00186217-340')
      - year: год выпуска стандарта (берём последний валидный год 1900–2099), тип Int64 (nullable)
    """
    if col not in df.columns:
        return df

    sys_list, code_list, year_list = [], [], []

    for raw in df[col].fillna(""):
        text = str(raw).strip()
        # Нормализация тире и пробелов
        t = (text.replace("–", "-").replace("—", "-")
                    .replace("\u00A0", " "))  # NBSP -> space

        # 1) system — первое буквенное слово в начале, иначе ищем известные системы внутри
        m_sys = re.match(r"^\s*([A-Za-zА-Яа-я]+)", t)
        system = m_sys.group(1) if m_sys else None
        if not system:
            m_known = re.search(r"\b(ГОСТ|EN|ISO|DIN|ASTM|СТО|ТУ)\b", t, flags=re.IGNORECASE)
            system = m_known.group(1) if m_known else "UNKNOWN"
        system = system.upper() if re.match(r"^[A-Za-z]+$", system) else system

        # 2) year — последний валидный год (1900–2099)
        year_matches = list(re.finditer(r"(?<!\d)(19|20)\d{2}(?!\d)", t))
        if year_matches:
            year = int(year_matches[-1].group(0))
            year_pos = year_matches[-1].start()
        else:
            year = pd.NA
            year_pos = len(t)

        # 3) code — часть между системой и годом
        start = m_sys.end() if m_sys else 0
        candidate = t[start:year_pos]

        # удаляем всё после маркеров изменений внутри «кода»
        candidate = re.sub(r"\b(изм|измен|попр|amend|amd)\b.*$", "", candidate, flags=re.IGNORECASE)

        # чистим символы разделителей с хвоста/начала
        candidate = candidate.strip(" :,-;/\t")
        candidate = re.sub(r"[:\s-]+$", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate)

        # финальные значения
        sys_list.append(system if system else "UNKNOWN")
        code_list.append(candidate if candidate else "UNKNOWN")
        year_list.append(year)

    out = df.copy()
    out["system"] = sys_list
    out["code"] = code_list
    out["year"] = pd.Series(year_list, dtype="Int64")
    return out




# --- Определение типов колонок ---
def detect_column_types(df):
    numeric_cols, categorical_cols = [], []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # пробуем привести к числу
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() > 0 and converted.isna().sum() == 0:
                numeric_cols.append(col)
                df[col] = converted  # конвертируем в число
            else:
                categorical_cols.append(col)

    return df, numeric_cols, categorical_cols


# --- Подготовка признаков ---
# def prepare_features_with_standard(df, selected_cols, target_col=None):
#     df = df.copy()

#     # отделяем таргет
#     y = None
#     if target_col and target_col in df.columns:
#         y = df[target_col].values
#         df = df.drop(columns=[target_col])

#     # работаем только с выбранными колонками
#     df = df[selected_cols]

#     # обработка пропусков
#     filled_cols = []
#     for col in df.columns:
#         if df[col].isna().any():
#             filled_cols.append(col)
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col] = df[col].fillna(0)
#             else:
#                 df[col] = df[col].fillna("UNKNOWN")

#     # определяем числовые и категориальные признаки
#     df, numeric_cols, categorical_cols = detect_column_types(df)

#     transformers = []

#     # числовые признаки → StandardScaler
#     if numeric_cols:
#         transformers.append(('num', StandardScaler(), numeric_cols))

#     # system → OneHot
#     if 'system' in df.columns:
#         transformers.append(('system', OneHotEncoder(handle_unknown='ignore', sparse_output=True), ['system']))

#     # code → TF-IDF
#     if 'code' in df.columns:
#         transformers.append(('code', TfidfVectorizer(analyzer='char', ngram_range=(3, 4)), 'code'))

#     # year → StandardScaler
#     if 'year' in df.columns:
#         transformers.append(('year', StandardScaler(), ['year']))

#     # has_updates → passthrough
#     if 'has_updates' in df.columns:
#         transformers.append(('has_updates', 'passthrough', ['has_updates']))

#     # MARKA → TF-IDF
#     if 'MARKA' in df.columns:
#         transformers.append(('marka', TfidfVectorizer(analyzer='char', ngram_range=(3, 4)), 'MARKA'))

#     # прочие категориальные → OneHot
#     other_cats = [c for c in categorical_cols if c not in ['system', 'code', 'year', 'has_updates', 'MARKA']]
#     if other_cats:
#         transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), other_cats))

#     preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

#     X = preprocessor.fit_transform(df)
#     return X, y, filled_cols

def prepare_features_with_standard(df, selected_cols, target_col=None):
    """
    Подготовка данных для кластеризации:
    - Числовые признаки → StandardScaler
    - Категориальные признаки:
        * если уникальных значений ≤10 → OneHotEncoder
        * если уникальных значений >10 → LabelEncoder
    - Текстовые признаки ('code', 'MARKA') → TF-IDF
    """

    df = df.copy()

    # --- Отделяем таргет
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values
        df = df.drop(columns=[target_col])

    df = df[selected_cols]

    # --- Обработка пропусков
    filled_cols = []
    for col in df.columns:
        if df[col].isna().any():
            filled_cols.append(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("UNKNOWN")

    # --- Определяем типы признаков
    df, numeric_cols, categorical_cols = detect_column_types(df)

    transformers = []
    log_messages = []  # для вывода в интерфейсе

    # --- Числовые признаки → StandardScaler
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
        log_messages.append(f"🔹 Числовые признаки ({len(numeric_cols)}): {', '.join(numeric_cols)}")

    # --- Категориальные признаки → OHE или LabelEncoder
    le_encoded = []
    for col in categorical_cols:
        # ⚠️ Пропускаем колонки, для которых будет отдельная TF-IDF обработка
        if col in ["code", "MARKA"]:
            continue

        n_unique = df[col].nunique()
        if n_unique <= 10:
            transformers.append((f"ohe_{col}", OneHotEncoder(handle_unknown="ignore"), [col]))
            log_messages.append(f"🟢 {col} (OHE, {n_unique} уникальных)")
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            transformers.append((f"le_{col}", "passthrough", [col]))
            le_encoded.append(col)
            log_messages.append(f"🟠 {col} (LabelEncoder, {n_unique} уникальных)")


    # --- TF-IDF для текстовых признаков
   
    for text_col in ["code", "MARKA"]:
        if text_col in df.columns:
            df[text_col] = df[text_col].astype(str)

    if "code" in df.columns:
        transformers.append(("code_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 4)), "code"))
        log_messages.append("🔤 'code' → TF-IDF")

    if "MARKA" in df.columns:
        transformers.append(("marka_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 4)), "MARKA"))
        log_messages.append("🔤 'MARKA' → TF-IDF")

    # --- ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    X = preprocessor.fit_transform(df)
    
    # --- Получаем имена признаков после трансформации
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X, y, filled_cols, log_messages, feature_names

