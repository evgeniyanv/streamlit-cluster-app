import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, davies_bouldin_score

from preprocessing import (
    clean_dataset,
    remove_outliers_iqr,
    detect_column_types,
    add_standard_features_v2,
    prepare_features_with_standard
)

# пробуем импортировать FAISS
try:
    import faiss
    faiss_available = True
except ImportError:
    faiss_available = False


# ---------- Интерфейс ----------
st.title("🔎 Кластеризация марок стали")
st.markdown("Загрузите датасет, выполните очистку, выберите признаки и алгоритм кластеризации.")

# --- Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV или XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("📊 Исходные данные:")
    st.dataframe(df.head())

    before_rows = len(df)

    # --- Чекбоксы для очистки
    remove_dupes = st.checkbox("Удалить дубликаты строк", value=True)
    remove_nans = st.checkbox("Удалить строки с пропусками", value=True)
    remove_outliers = st.checkbox("Удалить выбросы по IQR", value=False)

    if remove_dupes:
        df = df.drop_duplicates()
    if remove_nans:
        df = clean_dataset(df)
    if remove_outliers:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df = remove_outliers_iqr(df, numeric_cols)

    after_rows = len(df)
    st.success(f"✅ Очистка завершена. Было: {before_rows}, стало: {after_rows}")

    # --- Обработка ST_PROD ДО выбора признаков
    if "ST_PROD" in df.columns:
        df = add_standard_features_v2(df, col="ST_PROD")
    else:
        st.info("ℹ️ Колонка 'ST_PROD' не найдена — признаки system, code, year, has_updates не будут сгенерированы.")

    # --- Все колонки
    all_cols = list(df.columns)

    # --- Исключаем исходный столбец ST_PROD, но оставляем распарсенные признаки
    exclude_cols = ["ST_PROD"]  

    # --- Выбор таргета
    target_col = st.selectbox("🎯 Выберите таргет (опционально)", [None] + all_cols)
    if target_col:
        exclude_cols.append(target_col)

    # --- Доступные признаки
    available_cols = [c for c in all_cols if c not in exclude_cols]

    # --- Выбор признаков
    selected_cols = st.multiselect(
        "🧩 Признаки для кластеризации",
        available_cols,
        default=available_cols
    )

    st.info(f"✅ Выбрано признаков: {len(selected_cols)}")

    # --- Автоопределение типов
    if selected_cols:
        df_checked, numeric_cols, categorical_cols = detect_column_types(df[selected_cols].copy())
        st.subheader("🔎 Автоматическое определение типов признаков")
        st.markdown(f"**Числовые признаки ({len(numeric_cols)}):** {', '.join(numeric_cols) if numeric_cols else 'нет'}")
        st.markdown(f"**Категориальные признаки ({len(categorical_cols)}):** {', '.join(categorical_cols) if categorical_cols else 'нет'}")

    # --- Служебные признаки (в отдельном блоке)
    with st.expander("🔧 Служебные признаки (для проверки)"):
        service_cols = [c for c in ["system", "code", "year", "has_updates"] if c in df.columns]
        if service_cols:
            st.dataframe(df[service_cols].head())
        else:
            st.write("Нет служебных признаков.")

    # --- Алгоритм кластеризации
    
    n_rows = len(df)

    algo_options = []
    algo_info = []

    # KMeans всегда
    algo_options.append("KMeans (до ~100k строк)")

    # Agglomerative
    if n_rows <= 20000:
        algo_options.append("Agglomerative (до ~10k строк, иерархический)")
    else:
        algo_info.append("❌ Agglomerative недоступен для датасетов >20k строк (слишком ресурсоёмкий).")

    # DBSCAN
    if n_rows <= 50000:
        algo_options.append("DBSCAN (до ~50k строк, плотные кластеры)")
    else:
        algo_info.append("❌ DBSCAN недоступен для датасетов >50k строк (медленно и много памяти).")

    # FAISS
    if faiss_available:
        algo_options.append("FAISS HNSW (миллионы строк, быстрый поиск ближайших соседей)")
    else:
        algo_info.append("❌ FAISS недоступен — установите пакет `faiss-cpu`.")

    # Доп. инфо при больших данных
    if n_rows > 100000:
        algo_info.append("⚠️ Для датасетов >100k строк используется PCA для визуализации вместо UMAP/t-SNE.")

    # Selectbox
    algo = st.selectbox("🤖 Алгоритм кластеризации", algo_options)

    # Пояснения
    if algo_info:
        st.markdown("### ℹ️ Ограничения по выбору алгоритмов")
        for msg in algo_info:
            st.markdown(msg)

    # --- Параметры
    n_clusters, eps, min_samples, M = None, None, None, None
    if "KMeans" in algo:
        n_clusters = st.slider("Количество кластеров (KMeans)", 2, 20, 5)
    elif "Agglomerative" in algo:
        n_clusters = st.slider("Количество кластеров (Agglomerative)", 2, 15, 5)
    elif "DBSCAN" in algo:
        eps = st.slider("eps (радиус соседства, DBSCAN)", 0.1, 10.0, 0.5, step=0.1)
        min_samples = st.slider("min_samples (минимум точек в кластере, DBSCAN)", 2, 20, 5)
    elif "FAISS" in algo:
        n_clusters = st.slider("Количество кластеров (FAISS HNSW)", 2, 100, 10)
        M = st.slider("M (размер графа HNSW)", 8, 64, 32)

    # --- Запуск
    if st.button("🚀 Запустить кластеризацию"):
        X, y, filled_cols, log_messages, feature_names = prepare_features_with_standard(df, selected_cols, target_col)
        # Выводим лог предобработки
        with st.expander("📋 Лог предобработки признаков"):
            for msg in log_messages:
                st.markdown(msg)

        if filled_cols:
            st.warning(f"⚠ Заполнены пропуски в: {', '.join(filled_cols)}")

        if X.shape[0] == 0:
            st.error("Нет признаков для кластеризации!")
        else:
            if "KMeans" in algo:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                clusters = model.fit_predict(X)
            elif "Agglomerative" in algo:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                clusters = model.fit_predict(X.toarray() if hasattr(X, "toarray") else X)
            elif "DBSCAN" in algo:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(X.toarray() if hasattr(X, "toarray") else X)
            elif "FAISS" in algo:
                if not faiss_available:
                    st.error("⚠ FAISS не установлен! Установите: pip install faiss-cpu")
                    clusters = np.zeros(X.shape[0], dtype=int)
                else:
                    X_f32 = X.astype(np.float32).toarray() if hasattr(X, "toarray") else X.astype(np.float32)
                    d = X_f32.shape[1]
                    index = faiss.IndexHNSWFlat(d, M)
                    index.hnsw.efConstruction, index.hnsw.efSearch = 200, 50
                    index.add(X_f32)
                    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False)
                    kmeans.train(X_f32)
                    _, clusters = kmeans.index.search(X_f32, 1)
                    clusters = clusters.flatten()

            df["cluster"] = clusters

            # --- Метрики
            st.subheader("📊 Метрики качества")
            valid_clusters = set(clusters)
            if len(valid_clusters) > 1 and not (-1 in valid_clusters and "DBSCAN" in algo):
                sample_size = min(5000, X.shape[0])
                if X.shape[0] > sample_size:
                    idx = np.random.choice(X.shape[0], sample_size, replace=False)
                    X_eval, clusters_eval = X[idx], clusters[idx]
                else:
                    X_eval, clusters_eval = X, clusters

                sil_score = silhouette_score(X_eval, clusters_eval)
                db_score = davies_bouldin_score(X_eval, clusters_eval)

                st.success(f"Silhouette Score: **{sil_score:.3f}** (ближе к 1 лучше)")
                st.info(f"Davies–Bouldin Index: **{db_score:.3f}** (ближе к 0 лучше)")
            else:
                st.warning("⚠ Метрики не рассчитаны (слишком мало кластеров или есть шумовые метки).")

            # --- Барчарт
            st.subheader("📊 Размеры кластеров")
            cluster_sizes = df["cluster"].value_counts().reset_index()
            cluster_sizes.columns = ["cluster", "size"]
            fig_bar = px.bar(cluster_sizes, x="cluster", y="size", text="size",
                            title="Размеры кластеров", color="cluster")
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- Визуализация кластеров ---
            st.subheader("🌐 Визуализация кластеров")

            # Ограничиваем количество точек для визуализации
            max_viz_samples = 5000
            if len(X) > max_viz_samples:
                viz_idx = np.random.choice(len(X), max_viz_samples, replace=False)
                X_viz = X[viz_idx]
                clusters_viz = clusters[viz_idx]
                df_viz = df.iloc[viz_idx].copy()
                st.info(f"⚠️ Для визуализации использована подвыборка из {max_viz_samples} строк (всего {len(X)}).")
            else:
                X_viz = X
                clusters_viz = clusters
                df_viz = df.copy()

            # --- Снижение размерности 2D
            try:
                if len(X_viz) <= 5000:
                    reducer_2d = TSNE(n_components=2, random_state=42, perplexity=30)
                    X_embedded_2d = reducer_2d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                    method_2d = "t-SNE"
                else:
                    reducer_2d = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42, low_memory=True, n_components=2)
                    X_embedded_2d = reducer_2d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                    method_2d = "UMAP"
            except Exception:
                from sklearn.decomposition import PCA
                reducer_2d = PCA(n_components=2, random_state=42)
                X_embedded_2d = reducer_2d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                method_2d = "PCA"

            viz_df_2d = pd.DataFrame({
                "x": X_embedded_2d[:, 0],
                "y": X_embedded_2d[:, 1],
                "cluster": clusters_viz.astype(int)
            })


            viz_df_2d["cluster"] = viz_df_2d["cluster"].astype(str)

            # Добавим первые 3 исходных признака для отображения при наведении
            for col in selected_cols[:3]:
                if col in df.columns:
                    viz_df_2d[col] = df[col].iloc[:len(viz_df_2d)].values

            fig2d = px.scatter(
                viz_df_2d,
                x="x", y="y",
                color="cluster",
                hover_data=selected_cols[:3],
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"2D визуализация кластеров ({method_2d}, {algo})"
            )

            st.plotly_chart(fig2d, use_container_width=True)


            # --- Снижение размерности 3D (t-SNE / UMAP / PCA по ограничению размера)
            try:
                if len(X_viz) <= 10000:
                    reducer_3d = TSNE(n_components=3, random_state=42, perplexity=30)
                    X_embedded_3d = reducer_3d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                    method_3d = "t-SNE 3D"
                elif len(X_viz) <= 150000:
                    reducer_3d = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42, low_memory=True, n_components=3)
                    X_embedded_3d = reducer_3d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                    method_3d = "UMAP 3D"
                else:
                    reducer_3d = PCA(n_components=3, random_state=42)
                    X_embedded_3d = reducer_3d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                    method_3d = "PCA 3D"
            except Exception:
                reducer_3d = PCA(n_components=3, random_state=42)
                X_embedded_3d = reducer_3d.fit_transform(X_viz.toarray() if hasattr(X_viz, "toarray") else X_viz)
                method_3d = "PCA 3D (fallback)"

            viz_df_3d = pd.DataFrame({
                "x": X_embedded_3d[:, 0],
                "y": X_embedded_3d[:, 1],
                "z": X_embedded_3d[:, 2],
                "cluster": clusters_viz.astype(int)
            })

            fig3d = go.Figure(data=go.Scatter3d(
                x=viz_df_3d["x"], y=viz_df_3d["y"], z=viz_df_3d["z"],
                mode="markers",
                marker=dict(
                    color=viz_df_3d["cluster"],
                    colorscale="Rainbow",
                    size=2,
                    opacity=0.7,
                    colorbar=dict(title="Cluster")
                ),
                text=[f"Cluster: {c}" for c in viz_df_3d["cluster"]]
            ))
            fig3d.update_layout(title=f"3D визуализация кластеров ({method_3d}, {algo})")
            st.plotly_chart(fig3d, use_container_width=True)

            # --- Дендрограмма для Agglomerative

            def plot_dendrogram(X, method="ward", max_samples=500):
                # если данных очень много, берём подвыборку
                if X.shape[0] > max_samples:
                    idx = np.random.choice(X.shape[0], max_samples, replace=False)
                    X_sample = X[idx]
                else:
                    X_sample = X

                # linkage matrix
                Z = linkage(X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample, method=method)

                # строим график
                fig, ax = plt.subplots(figsize=(12, 6))
                dendrogram(Z, ax=ax, truncate_mode="level", p=10)  # p=10 → глубина отображаемых уровней
                ax.set_title("Дендрограмма (иерархическая кластеризация)")
                ax.set_xlabel("Объекты или кластеры")
                ax.set_ylabel("Расстояние")
                st.pyplot(fig)

            if "Agglomerative" in algo:
                st.subheader("🌳 Дендрограмма для Agglomerative Clustering")
                plot_dendrogram(X, method="ward")

 
            # --- Анализ ключевых признаков кластеров ---

            st.subheader("🧩 Анализ ключевых признаков кластеров")

            try:
                X_rf = X
                y_rf = df["cluster"].astype(str)

                if hasattr(X_rf, "toarray"):
                    X_rf = X_rf.toarray()

                rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                rf.fit(X_rf, y_rf)

                # создаём DataFrame с реальными именами признаков
                importances_df = pd.DataFrame({
                    "Признак": feature_names,
                    "Важность": rf.feature_importances_
                }).sort_values(by="Важность", ascending=False)

                top_features = importances_df.head(5)

                st.success("✅ Random Forest обучен. Ниже — 5 признаков, наиболее влияющих на разбиение на кластеры.")
                st.table(top_features.round(4))

                fig_importance = px.bar(
                    top_features[::-1],
                    x="Важность", y="Признак",
                    orientation="h",
                    title="Топ-5 признаков по важности (Random Forest)",
                    color="Важность",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_importance, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ Не удалось рассчитать важность признаков: {e}")


            # --- Таблица
            st.subheader("📋 Примеры данных из кластеров")
            st.dataframe(df[selected_cols + ["cluster"]].head(10))
            
            # --- Кнопки для скачивания результатов ---
            st.subheader("💾 Скачать результаты кластеризации")

            # CSV — всегда
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать CSV",
                data=csv_data,
                file_name="clusters.csv",
                mime="text/csv"
            )

            # Excel — только для небольших датасетов
            if len(df) <= 100000:
                excel_file = "clusters.xlsx"
                df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button(
                        label="Скачать Excel",
                        data=f,
                        file_name="clusters.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.info("ℹ️ Excel-файл не предлагается для датасетов >100k строк, "
                        "так как он будет слишком тяжёлым. Используйте CSV.")
