"""銅箔製程 Streamlit 主程式。

啟動：python -m streamlit run 05_app.py
視覺化使用原始單位的清理資料；推論沿用 03／04 的模型與固定前處理。
所有路徑以本程式位置為基準，方便從不同目錄啟動或部署。
"""

from datetime import datetime, time, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_preparation" / "copper_foil_cleaned.csv"
MODEL_DIR = BASE_DIR / "random_forest_training"
MODEL_PATH = MODEL_DIR / "random_forest_anomaly.joblib"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
PARAMETERS_PATH = MODEL_DIR / "preprocessing_parameters.json"
TAIPEI = timezone(timedelta(hours=8))

# 輸入與圖表均使用原始製程單位，使用者不必自行計算 z 分數。
FEATURE_LABELS = {
    "Line_Speed_mpm": "線速度（m/min）",
    "Current_ASD": "電流密度（A/dm²）",
    "Bath_Temp_C": "槽液溫度（°C）",
    "Cu_Conc_gL": "銅濃度（g/L）",
    "Additive_ppm": "添加劑濃度（ppm）",
    "Jet_Pressure_bar": "噴流壓力（bar）",
    "Edge_Current_Bias": "邊緣電流偏差",
}
PROCESS_COLS = list(FEATURE_LABELS)
MAIN_FEATURES = ["Current_ASD", "Edge_Current_Bias", "Line_Speed_mpm", "Jet_Pressure_bar"]
QUALITY_COLORS = alt.Scale(domain=["正常", "異常"], range=["#287b9b", "#d96945"])


def file_signature(path: Path) -> tuple[int, int]:
    """檔案更新時使快取失效；缺檔交由頁面顯示說明。"""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def load_dashboard_data(path: str, signature: tuple[int, int]) -> pd.DataFrame:
    """快取清理資料，避免切換控制項時重複讀取 CSV。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["Sample_ID", "Timestamp", "Shift", "Is_Anomaly"] + PROCESS_COLS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"視覺化資料缺少欄位：{missing}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("清理資料為空或仍有缺失值，請重新執行 02 notebook。")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
    if frame["Timestamp"].isna().any():
        raise ValueError("視覺化資料包含無效時間。")
    for column in PROCESS_COLS + ["Is_Anomaly"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[PROCESS_COLS + ["Is_Anomaly"]].to_numpy(dtype=float)).all():
        raise ValueError("視覺化資料包含無限值或無效數值。")
    if not frame["Is_Anomaly"].isin([0, 1]).all():
        raise ValueError("Is_Anomaly 必須為 0 或 1。")
    if not frame["Shift"].isin(["早班", "中班", "晚班"]).all():
        raise ValueError("班別必須為早班、中班或晚班。")
    frame["品質類別"] = frame["Is_Anomaly"].map({0: "正常", 1: "異常"})
    return frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)


@st.cache_resource(show_spinner="正在載入預測模型…")
def load_prediction_bundle(model_path: str, metadata_path: str, parameter_path: str, signatures: tuple) -> dict:
    """載入已知來源的部署模型，驗證參數配對並快取已擬合 Pipeline。"""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    model_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    if model_hash != metadata["model_sha256"]:
        raise ValueError("模型與 training_metadata.json 不屬於同一次訓練，請重新匯出。")
    # joblib 對環境版本敏感；部署時使用隨附 requirements.txt。
    for package in ["scikit-learn", "numpy", "pandas", "joblib"]:
        if version(package) != metadata["versions"][package]:
            raise ValueError(f"{package} 版本與訓練環境不同，請安裝 requirements.txt 的版本。")
    preprocessing = metadata.get("preprocessing_parameters")
    if preprocessing is None:
        preprocessing = json.loads(Path(parameter_path).read_text(encoding="utf-8"))
    if (metadata["feature_columns"] != preprocessing["feature_columns"]
            or metadata["process_columns"] != preprocessing["process_columns"]
            or metadata["target_column"] != preprocessing["target_column"]):
        raise ValueError("模型與前處理的特徵／目標設定不一致。")
    if metadata["process_columns"] != PROCESS_COLS:
        raise ValueError("模型製程欄位與本表單定義不同。")
    if preprocessing["shift_categories"] != ["早班", "中班", "晚班"]:
        raise ValueError("模型班別定義與本程式不同。")
    cycles = preprocessing["time_cycles"]
    if cycles["weekday_zero"] != "Monday" or cycles["Hour"] != 24 or cycles["DayOfWeek"] != 7:
        raise ValueError("週期時間定義與本程式不同。")
    threshold = float(metadata["decision_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("分類門檻必須介於 0 與 1。")
    means = np.array([preprocessing["scaler_mean"][col] for col in PROCESS_COLS], dtype=float)
    scales = np.array([preprocessing["scaler_scale"][col] for col in PROCESS_COLS], dtype=float)
    medians = np.array([preprocessing["train_medians"][col] for col in PROCESS_COLS], dtype=float)
    if (not np.isfinite(means).all() or not np.isfinite(scales).all()
            or not np.isfinite(medians).all() or (scales <= 0).any()):
        raise ValueError("前處理數值參數無效。")
    model = joblib.load(model_path)
    if (list(model.feature_names_in_) != metadata["feature_columns"]
            or model.n_features_in_ != len(metadata["feature_columns"])
            or set(model.classes_) != {0, 1}):
        raise ValueError("模型輸入欄位或分類標籤與訓練紀錄不一致。")
    return {"model": model, "metadata": metadata, "preprocessing": preprocessing,
            "threshold": threshold, "model_sha256": model_hash}


def transform_raw_to_model_input(raw_frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """沿用 04：固定標準化、班別編碼、週期時間及缺失指標。"""
    missing = sorted(set(["Timestamp"] + PROCESS_COLS) - set(raw_frame.columns))
    if missing:
        raise ValueError(f"缺少輸入欄位：{missing}")
    if raw_frame.empty or raw_frame.columns.duplicated().any():
        raise ValueError("輸入不可為空，欄位名稱不可重複。")
    timestamps = pd.to_datetime(raw_frame["Timestamp"], errors="raise")
    if timestamps.isna().any() or timestamps.dt.tz is not None:
        raise ValueError("請提供未附時區的台北當地量測時間。")
    preprocessing = bundle["preprocessing"]
    process = raw_frame[PROCESS_COLS].apply(pd.to_numeric, errors="coerce")
    process = process.replace([np.inf, -np.inf], np.nan).astype(float)
    positive_cols = [col for col in PROCESS_COLS if col not in ["Bath_Temp_C", "Edge_Current_Bias"]]
    process[positive_cols] = process[positive_cols].mask(process[positive_cols].le(0))
    missing_flags = process.isna().astype("int8").add_suffix("_was_missing")
    means = pd.Series(preprocessing["scaler_mean"]).reindex(PROCESS_COLS)
    scales = pd.Series(preprocessing["scaler_scale"]).reindex(PROCESS_COLS)
    # 不重新 fit scaler；NaN 保留給 joblib 內已擬合的 SimpleImputer。
    scaled = ((process - means) / scales).add_suffix("_z")
    hours = timestamps.dt.hour
    shift = pd.Series(np.select(
        [hours.between(8, 15), hours.between(16, 23)], ["早班", "中班"], default="晚班",
    ), index=raw_frame.index)
    shifts = pd.DataFrame({
        f"Shift_{category}": shift.eq(category).astype("int8")
        for category in preprocessing["shift_categories"]
    }, index=raw_frame.index)
    hour_fraction = hours + timestamps.dt.minute / 60
    weekday = timestamps.dt.dayofweek
    cycles = preprocessing["time_cycles"]
    time_features = pd.DataFrame({
        "Hour_sin": np.sin(2 * np.pi * hour_fraction / cycles["Hour"]),
        "Hour_cos": np.cos(2 * np.pi * hour_fraction / cycles["Hour"]),
        "DayOfWeek_sin": np.sin(2 * np.pi * weekday / cycles["DayOfWeek"]),
        "DayOfWeek_cos": np.cos(2 * np.pi * weekday / cycles["DayOfWeek"]),
    }, index=raw_frame.index)
    features = pd.concat([scaled, shifts, time_features, missing_flags], axis=1)
    feature_columns = bundle["metadata"]["feature_columns"]
    if set(features.columns) != set(feature_columns):
        raise ValueError("轉換後特徵與模型定義不一致。")
    features = features.loc[:, feature_columns].astype(float)
    if np.isinf(features.to_numpy()).any():
        raise ValueError("轉換後出現無限值，請確認輸入。")
    return features


def predict_process(raw_frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """使用保存的分類門檻，輸出機率及標籤，不重新訓練模型。"""
    features = transform_raw_to_model_input(raw_frame, bundle)
    model = bundle["model"]
    positive_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    probabilities = model.predict_proba(features)[:, positive_index]
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("模型未輸出有效機率。")
    result = raw_frame.copy()
    result["Anomaly_Probability"] = probabilities
    result["Predicted_Is_Anomaly"] = (probabilities >= bundle["threshold"]).astype("int8")
    result["Predicted_Label"] = result["Predicted_Is_Anomaly"].map({0: "正常", 1: "異常"})
    result["Input_Missing_Count"] = features.isna().sum(axis=1)
    return result


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """含 BOM 的 UTF-8 CSV，便於 Excel 顯示中文。"""
    return frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M").encode("utf-8-sig")


def quality_color() -> alt.Color:
    return alt.Color("品質類別:N", scale=QUALITY_COLORS, title="歷史品質")


def render_dashboard() -> None:
    st.title("製程資料視覺化")
    st.caption("依日期與班別探索製程條件；圖中的正常／異常來自歷史品質標籤。")
    try:
        frame = load_dashboard_data(str(DATA_PATH), file_signature(DATA_PATH))
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"無法讀取視覺化資料：{error}")
        st.info("請將 data_preparation/copper_foil_cleaned.csv 一併部署，或先執行 02 notebook。")
        return

    st.sidebar.divider()
    st.sidebar.subheader("資料篩選")
    first_day, last_day = frame["Timestamp"].min().date(), frame["Timestamp"].max().date()
    date_range = st.sidebar.date_input("日期範圍", value=(first_day, last_day),
                                       min_value=first_day, max_value=last_day, key="date_range")
    shifts = st.sidebar.multiselect("班別", ["早班", "中班", "晚班"],
                                    default=["早班", "中班", "晚班"], key="shift_filter")
    labels = st.sidebar.multiselect("歷史品質", ["正常", "異常"],
                                    default=["正常", "異常"], key="quality_filter")
    if len(date_range) != 2:
        st.info("請選擇起始與結束日期。")
        return
    start, end = date_range
    filtered = frame.loc[frame["Timestamp"].dt.date.between(start, end)
                         & frame["Shift"].isin(shifts) & frame["品質類別"].isin(labels)].copy()
    if filtered.empty:
        st.info("此篩選條件沒有資料，請調整日期、班別或歷史品質。")
        return

    metrics = st.columns(4)
    metrics[0].metric("樣本筆數", f"{len(filtered):,}", border=True)
    metrics[1].metric("歷史異常筆數", f"{int(filtered['Is_Anomaly'].sum()):,}", border=True)
    metrics[2].metric("歷史異常比例", f"{filtered['Is_Anomaly'].mean():.1%}", border=True)
    metrics[3].metric("涵蓋天數", f"{filtered['Timestamp'].dt.date.nunique()}", border=True)
    features = st.multiselect("顯示主要製程特徵", PROCESS_COLS, default=MAIN_FEATURES,
                             format_func=FEATURE_LABELS.get, key="chart_features")
    if not features:
        st.info("請至少選擇一個製程特徵。")
        return
    trend_tab, distribution_tab, relation_tab, detail_tab = st.tabs(
        ["製程趨勢", "特徵分布", "特徵關聯", "資料明細"]
    )
    with trend_tab:
        chart_columns = st.columns(2)
        for index, feature in enumerate(features):
            # 趨勢線依完整時間順序連接；散點依歷史標籤著色。
            base = alt.Chart(filtered)
            line = base.mark_line(color="#87a5b2", strokeWidth=1).encode(
                x=alt.X("Timestamp:T", title="量測時間"),
                y=alt.Y(f"{feature}:Q", title=FEATURE_LABELS[feature], scale=alt.Scale(zero=False)),
            )
            points = base.mark_circle(size=18, opacity=0.75).encode(
                x="Timestamp:T", y=f"{feature}:Q", color=quality_color(),
                tooltip=["Sample_ID:N", "Timestamp:T", alt.Tooltip(f"{feature}:Q", format=".2f"), "品質類別:N"],
            )
            chart_columns[index % 2].altair_chart(
                (line + points).properties(title=FEATURE_LABELS[feature], height=230).interactive(),
                width="stretch",
            )
        shift_chart = alt.Chart(filtered).mark_bar().encode(
            x=alt.X("Shift:N", title="班別", sort=["早班", "中班", "晚班"]),
            y=alt.Y("count():Q", title="樣本筆數"), color=quality_color(),
            tooltip=["Shift:N", "品質類別:N", alt.Tooltip("count():Q", title="筆數")],
        ).properties(title="各班別歷史品質分布", height=250)
        st.altair_chart(shift_chart, width="stretch")

    with distribution_tab:
        feature = st.selectbox("分布特徵", features, format_func=FEATURE_LABELS.get, key="distribution_feature")
        left, right = st.columns(2)
        histogram = alt.Chart(filtered).mark_bar(opacity=0.65).encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=FEATURE_LABELS[feature]),
            y=alt.Y("count():Q", title="筆數", stack=None), color=quality_color(),
            tooltip=["品質類別:N", alt.Tooltip("count():Q", title="筆數")],
        ).properties(title="特徵直方圖", height=320)
        boxplot = alt.Chart(filtered).mark_boxplot(extent=1.5).encode(
            x=alt.X("品質類別:N", title="歷史品質", sort=["正常", "異常"]),
            y=alt.Y(f"{feature}:Q", title=FEATURE_LABELS[feature], scale=alt.Scale(zero=False)),
            color=quality_color(),
        ).properties(title="正常／異常箱型圖", height=320)
        left.altair_chart(histogram, width="stretch")
        right.altair_chart(boxplot, width="stretch")

    with relation_tab:
        left, right = st.columns(2)
        x_feature = left.selectbox("散佈圖 X 軸", features, format_func=FEATURE_LABELS.get, key="scatter_x")
        y_feature = right.selectbox("散佈圖 Y 軸", features, index=min(1, len(features) - 1),
                                    format_func=FEATURE_LABELS.get, key="scatter_y")
        scatter = alt.Chart(filtered).mark_circle(size=45, opacity=0.65).encode(
            x=alt.X(f"{x_feature}:Q", title=FEATURE_LABELS[x_feature], scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y_feature}:Q", title=FEATURE_LABELS[y_feature], scale=alt.Scale(zero=False)),
            color=quality_color(), tooltip=["Sample_ID:N", "品質類別:N", x_feature, y_feature],
        ).properties(title="製程特徵散佈圖", height=340).interactive()
        st.altair_chart(scatter, width="stretch")
        correlations = filtered[features].rename(columns=FEATURE_LABELS).corr()
        long_corr = correlations.rename_axis("特徵 A").reset_index().melt(
            id_vars="特徵 A", var_name="特徵 B", value_name="相關係數",
        ).dropna(subset=["相關係數"])
        if long_corr.empty:
            st.info("目前樣本不足或特徵皆為固定值，無法計算相關係數。")
        else:
            base = alt.Chart(long_corr).encode(
                x=alt.X("特徵 A:N", title=None, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("特徵 B:N", title=None),
            )
            heatmap = base.mark_rect().encode(
                color=alt.Color("相關係數:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                tooltip=["特徵 A:N", "特徵 B:N", alt.Tooltip("相關係數:Q", format=".3f")],
            )
            numbers = base.mark_text().encode(
                text=alt.Text("相關係數:Q", format=".2f"),
                color=alt.condition("abs(datum['相關係數']) > 0.6", alt.value("white"), alt.value("#263238")),
            )
            st.altair_chart((heatmap + numbers).properties(title="Pearson 相關矩陣", height=340), width="stretch")
        st.caption("相關係數描述線性關係，不能據此判定因果。")

    with detail_tab:
        st.dataframe(filtered.drop(columns=["品質類別"]), hide_index=True, width="stretch")
        st.download_button("下載目前篩選資料", csv_bytes(filtered.drop(columns=["品質類別"])),
                           file_name="filtered_process_data.csv", mime="text/csv", key="download_filtered")
    st.caption("資料來源：教學用合成製程資料，圖表使用清理後的原始單位。")


def render_prediction() -> None:
    st.title("製程異常預測")
    st.caption("輸入量測時間與製程數值，取得 Random Forest 的異常機率及分類結果。")
    try:
        signatures = (file_signature(MODEL_PATH), file_signature(METADATA_PATH),
                      file_signature(PARAMETERS_PATH) if PARAMETERS_PATH.is_file() else None)
        bundle = load_prediction_bundle(str(MODEL_PATH), str(METADATA_PATH), str(PARAMETERS_PATH), signatures)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        st.error(f"無法載入預測模型：{error}")
        st.info("請確認 random_forest_training 目錄包含模型及同次訓練的 metadata，並安裝 requirements.txt。")
        return

    defaults = bundle["preprocessing"]["train_medians"]
    now = datetime.now(TAIPEI)
    with st.form("prediction_form"):
        st.subheader("量測資料")
        date_col, time_col = st.columns(2)
        measurement_date = date_col.date_input("量測日期（台北時間）", value=now.date(), key="measurement_date")
        measurement_time = time_col.time_input("量測時間", value=time(now.hour, now.minute // 15 * 15),
                                               step=timedelta(minutes=15), key="measurement_time")
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(PROCESS_COLS):
            # 邊緣偏差可為負數；其餘表單輸入皆需為正數。
            values[feature] = columns[index % 2].number_input(
                FEATURE_LABELS[feature], value=float(defaults[feature]),
                min_value=None if feature == "Edge_Current_Bias" else 0.01,
                step=0.01, format="%.2f", key=f"input_{feature}",
            )
        missing_columns = st.multiselect("未量測欄位（選填，使用既有模型補值）", PROCESS_COLS,
                                         format_func=FEATURE_LABELS.get, key="missing_columns")
        submitted = st.form_submit_button("進行異常預測", type="primary", width="stretch")

    if submitted:
        # 本次輸入無效時，清除上次結果，避免誤讀。
        st.session_state.pop("last_prediction", None)
        raw_input = pd.DataFrame([{
            "Timestamp": pd.Timestamp(datetime.combine(measurement_date, measurement_time)),
            **{feature: np.nan if feature in missing_columns else values[feature] for feature in PROCESS_COLS},
        }])
        try:
            with st.spinner("正在預測…"):
                result = predict_process(raw_input, bundle)
            st.session_state["last_prediction"] = {"result": result, "model_sha256": bundle["model_sha256"]}
        except (ValueError, TypeError, KeyError) as error:
            st.error(f"無法進行預測：{error}")

    saved = st.session_state.get("last_prediction")
    if saved and saved["model_sha256"] == bundle["model_sha256"]:
        st.divider()
        st.subheader("上次提交結果")
        result = saved["result"]
        row = result.iloc[0]
        probability = float(row["Anomaly_Probability"])
        if int(row["Predicted_Is_Anomaly"]) == 1:
            st.warning("預測結果：異常。建議確認製程條件及品質量測。")
        else:
            st.success("預測結果：正常。")
        metrics = st.columns(3)
        metrics[0].metric("異常機率分數", f"{probability:.1%}", border=True)
        metrics[1].metric("分類門檻", f"{bundle['threshold']:.0%}", border=True)
        metrics[2].metric("補值欄位數", f"{int(row['Input_Missing_Count'])}", border=True)
        st.progress(probability, text=f"異常機率分數 {probability:.1%}")
        st.caption(f"量測時間：{row['Timestamp']:%Y-%m-%d %H:%M}（台北時間）。結果對應已提交的輸入。")
        with st.expander("檢視已提交的資料與預測"):
            st.dataframe(result.rename(columns={**FEATURE_LABELS, "Timestamp": "量測時間",
                                                "Anomaly_Probability": "異常機率", "Predicted_Label": "預測類別",
                                                "Predicted_Is_Anomaly": "預測標籤", "Input_Missing_Count": "補值欄位數"}),
                         hide_index=True, width="stretch")
        st.download_button("下載此次預測結果", csv_bytes(result),
                           file_name="single_prediction_result.csv", mime="text/csv", key="download_prediction")
    else:
        st.info("請填寫表單並按下「進行異常預測」。")
    st.caption("模型以教學用合成資料訓練。異常分數未另做機率校準，預測仍需配合實際品質量測確認。")


def main() -> None:
    st.set_page_config(page_title="銅箔製程｜分析與異常預測", page_icon="🏭", layout="wide")
    st.sidebar.title("銅箔製程分析")
    st.sidebar.caption("Random Forest · 品質異常分類")
    page = st.sidebar.radio("功能分頁", ["資料視覺化", "異常預測"], key="page")
    if page == "資料視覺化":
        render_dashboard()
    else:
        render_prediction()


if __name__ == "__main__":
    main()
