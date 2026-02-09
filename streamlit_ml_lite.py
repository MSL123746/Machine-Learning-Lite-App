import streamlit as st
# New Repo Jan 8, 2026 - First Commit
st.markdown("""
    <style>
    /* Hide uploaded file list in file_uploader for all Streamlit versions */
    div[data-testid="stFileUploader"] ul,
    div[data-testid="stFileUploader"] .uploadedFile,
    div[data-testid="stFileUploader"] .st-emotion-cache-1m3b9l5,
    div[data-testid="stFileUploader"] .st-emotion-cache-1c7y2kd {
        display: none !important;
    }

    /* Modern card look for containers */
    section.main > div, .block-container, .stApp, .st-emotion-cache-1wrcr25 {
        background: #fff !important;
        border-radius: 18px !important;
        box-shadow: 0 2px 16px rgba(37,99,235,0.07) !important;
        padding: 2.2rem 2.5rem 2.5rem 2.5rem !important;
        margin-top: 1.2rem !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"], .st-emotion-cache-1v0mbdj, .stSidebar {
        background: #f8fafc !important;
        border-radius: 18px 0 0 18px !important;
        box-shadow: 2px 0 16px rgba(37,99,235,0.07) !important;
    }
    /* Sidebar button highlight */
    .mllite-stepper-btn.selected, .st-emotion-cache-1v0mbdj, .stSidebar .selected {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
        color: #fff !important;
        border: 1.5px solid #2563eb !important;
        border-left: 5px solid #2563eb !important;
        box-shadow: 0 4px 18px rgba(37,99,235,0.13) !important;
    }
    /* Sidebar button normal */
    .mllite-stepper-btn, .stSidebar button, .st-emotion-cache-1v0mbdj {
        border-radius: 10px !important;
        background: #fff !important;
        color: #1e293b !important;
        border: 1.5px solid #e5e7eb !important;
        box-shadow: 0 1px 6px rgba(37,99,235,0.07) !important;
    }
    /* File uploader styling */
    div[data-testid="stFileUploader"] > div:first-child {
        border-radius: 12px !important;
        background: #f1f5f9 !important;
        border: 1.5px solid #e5e7eb !important;
        box-shadow: 0 1px 6px rgba(37,99,235,0.07) !important;
    }
    /* Buttons */
    button, div.stButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
        color: #fff !important;
        font-weight: 600 !important;
        font-size: 1.08rem !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.10) !important;
        border: none !important;
        padding: 0.6em 2.2em !important;
        margin-bottom: 0.5em !important;
        transition: background 0.2s;
    }
    button:hover, div.stButton > button:hover {
        background: #1741a6 !important;
        color: #fff !important;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', 'Inter', Arial, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
    }
    /* General font */
    html, body, .stApp {
        font-family: 'Segoe UI', 'Inter', Arial, sans-serif !important;
        color: #1e293b !important;
    }
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 1.2rem !important;
    }
    /* Help button style */
    #help-fab {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
        color: #fff !important;
        border-radius: 50% !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.18) !important;
    }
    </style>
""", unsafe_allow_html=True)

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

def init_state():
    ss = st.session_state
    if 'step' not in ss:
        ss['step'] = 1
    defaults = {
        'algorithm': 'Linear Regression',
        'model_type': 'Regression',
        'uploaded_df': None,
        'df_sample': None,
        'schema': None,
        'target': None,
        'features': None,
        'settings': {},
        'trained_model': None,
        
        'metrics': None,
        'training_logs': [],
        'training_status': 'idle',
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v


# Ensure this function is defined before main block
def _run_with_streamlit_if_needed():
    """When the file is executed directly (e.g. via VS Code "Run Python File"),
    re-launch it under the Streamlit runner so the developer experience works
    without typing the long command.

    If the script is already running under Streamlit's script runner, just
    call main() normally.
    """
    import os
    import sys
    try:
        # If we're running under streamlit's runtime, get_script_run_ctx() will
        # return a context object. In that case, just execute main().
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    # Prevent accidental relaunch loops: if we already relaunched this process
    # into Streamlit once, don't try to exec again — just run main(). This can
    # happen if an external launcher triggers the script multiple times.
    if os.environ.get('STREAMLIT_RELAUNCHED') == '1':
        main()
        return

    if ctx is not None:
        # Running under streamlit already (e.g. `streamlit run ...`) — start app normally
        main()
    else:
        # Not running under streamlit: set a marker in the environment and
        # replace the current process with `python -m streamlit run <this file>`
        # so logs appear in the same terminal.
        os.environ['STREAMLIT_RELAUNCHED'] = '1'
        python = sys.executable or 'python'
        os.execv(python, [python, '-m', 'streamlit', 'run', __file__])

    def _run_with_streamlit_if_needed():
        """When the file is executed directly (e.g. via VS Code "Run Python File"),
        re-launch it under the Streamlit runner so the developer experience works
        without typing the long command.

        If the script is already running under Streamlit's script runner, just
        call main() normally.
        """
        import os
        import sys
        try:
            # If we're running under streamlit's runtime, get_script_run_ctx() will
            # return a context object. In that case, just execute main().
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None

        # Prevent accidental relaunch loops: if we already relaunched this process
        # into Streamlit once, don't try to exec again — just run main(). This can
        # happen if an external launcher triggers the script multiple times.
        if os.environ.get('STREAMLIT_RELAUNCHED') == '1':
            main()
            return

        if ctx is not None:
            # Running under streamlit already (e.g. `streamlit run ...`) — start app normally
            main()
        else:
            # Not running under streamlit: set a marker in the environment and
            # replace the current process with `python -m streamlit run <this file>`
            # so logs appear in the same terminal.
            os.environ['STREAMLIT_RELAUNCHED'] = '1'
            python = sys.executable or 'python'
            os.execv(python, [python, '-m', 'streamlit', 'run', __file__])

import io
import json
import pickle
import tempfile
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


st.set_page_config(page_title='Machine Learning Sandbox', layout='wide')
import sklearn

def main():
    from sklearn.linear_model import LogisticRegression
    st.write('LogisticRegression class:', LogisticRegression)
    init_state()
    # Place the title at the top left with custom styling
    st.markdown('<h1 style="text-align: left; margin-bottom: 0.5em;">Machine Learning Sandbox</h1>', unsafe_allow_html=True)
    sidebar_steps()
    ss = st.session_state
    step = ss.get('step', 1)
    if step == 1:
        step1_model_and_data()
    elif step == 2:
        step2_settings()
    elif step == 3:
        start_training()
        step3_training()
    elif step == 4:
        step4_results()
    # Add more steps as needed

def _inject_stepper_css():
    # CSS for sidebar and stepper
    css = """
    <style>
    .mllite-stepper-btn {
        display: flex;
        align-items: center;
        gap: 14px;
        width: 100%;
        background: #fff;
        border-radius: 10px;
        border: 1.5px solid #e5e7eb;
        box-shadow: 0 1px 6px rgba(37,99,235,0.07);
        padding: 0.7rem 1.1rem 0.7rem 0.9rem;
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.7rem;
        cursor: pointer;
        transition: box-shadow 0.13s, border 0.13s, background 0.13s, color 0.13s;
        outline: none;
        border-left: 5px solid transparent;
    }
    .mllite-stepper-btn:hover {
        box-shadow: 0 2px 12px rgba(37,99,235,0.13);
        border-color: #c7d2fe;
    }
    .mllite-stepper-btn.selected {
        background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
        color: #fff !important;
        border: 1.5px solid #2563eb !important;
        border-left: 5px solid #2563eb !important;
        box-shadow: 0 4px 18px rgba(37,99,235,0.13) !important;
    }
    .mllite-stepper-btn .step-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #f1f5f9;
        color: #64748b;
        font-size: 1.15rem;
        font-weight: 700;
        border: 2px solid #e5e7eb;
        transition: background 0.13s, color 0.13s, border 0.13s;
        flex-shrink: 0;
    }
    .mllite-stepper-btn.selected .step-circle {
        background: #2563eb;
        color: #fff;
        border: 2px solid #2563eb;
    }
    .mllite-stepper-btn .step-label {
        flex: 1;
        font-size: 1.07rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def infer_schema(df: pd.DataFrame):
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        # simple categorization
        if pd.api.types.is_numeric_dtype(df[col]):
            t = 'numeric'
        else:
            # treat as categorical if object or bool
            t = 'categorical'
        sample = df[col].dropna().head(3).tolist()
        schema.append({'column': col, 'type': t, 'dtype': dtype, 'sample': sample})
    return schema


def safe_pickle(obj):
    return pickle.dumps(obj)


def readable_exception(e: Exception):
    return f"Error: {str(e)}"





def sidebar_steps():
    # Inject CSS for prettier sidebar buttons
    _inject_stepper_css()
    ss = st.session_state
    ss = st.session_state
    if ss.get('model_type') == 'Clustering':
        steps = [
            ('Stage 1', 'Load Data', 'Upload CSV, choose target and features.'),
            ('Stage 2', 'Split Data', 'Select train fraction and algorithm settings.'),
            ('Stage 3', 'Training', 'Start training and view logs/progress.'),
            ('Stage 4', 'Results', 'Inspect metrics and download model.'),
        ]
    else:
        steps = [
            ('Stage 1', 'Load Data', 'Upload CSV, choose target and features.'),
            ('Stage 2', 'Split Data', 'Select train fraction and algorithm settings.'),
            ('Stage 3', 'Training', 'Start training and view logs/progress.'),
            ('Stage 4', 'Results', 'Inspect metrics and download model.'),
            ('Stage 5', 'Test Model', 'Make single or batch predictions.'),
        ]

    def set_step(idx):
        ss['step'] = idx + 1

    for i, (stage, label, desc) in enumerate(steps):
        btn_label = f"{stage}: {label}"
        selected = (ss.get('step', 1) == i + 1)
        if selected:
            st.sidebar.markdown(
                f"""
                <div style='background: #e0e7ff; border: 2px solid #2563eb; border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.7rem; font-weight: 600; font-size: 1.05rem; color: #1e293b; display: flex; align-items: center; min-height: 56px;'>
                    <span>{stage}: {label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.sidebar.button(
                btn_label,
                key=f"side_step_btn_{i}",
                help=desc,
                on_click=set_step,
                args=(i,)
            )
    st.sidebar.markdown("---")
    st.sidebar.success('ML App Ready for Use!')
    # Computer Vision button and related code removed
    # All computer vision, DummyClassifier, and test_gray code removed

def step1_model_and_data():
    st.header('Welcome to the Machine Learning SimulatorVersion 4')
    ss = st.session_state
    # ...existing code...
    col1, col2 = st.columns([2, 5])
    with col1:
        model_types = ['Regression', 'Binary classification', 'Multi-class classification', 'Clustering']
        mt = st.selectbox('Model Type', model_types, index=model_types.index(ss.get('model_type', 'Regression')))
        ss['model_type'] = mt
        st.markdown('Upload a CSV file (max 10 MB). The app will infer a simple schema.')
        csv_file = st.file_uploader('Upload CSV', type=['csv'])
        st.caption(':information_source: **Note:** The maximum file size for upload is 10 MB. If you see a higher limit, it is a Streamlit default, but this app enforces a 10 MB limit.')
        if csv_file is not None:
            try:
                data_bytes = csv_file.read()
                if len(data_bytes) > 10 * 1024 * 1024:
                    st.error('File too large (limit 10 MB).')
                    return
                df = pd.read_csv(io.BytesIO(data_bytes), na_values=['', ' '], keep_default_na=True)
                auto_cols = [c for c in df.columns if 'auto' in str(c).lower() or 'unique_id' in str(c).lower() or '::auto_unique_id::' in str(c)]
                if auto_cols:
                    df = df.drop(columns=auto_cols)
                st.session_state['uploaded_df'] = df
                # Immediately notify user if missing values are present in uploaded data
                missing_total = df.isna().sum().sum()
                if missing_total > 0:
                    st.warning(f"Missing values detected in uploaded data: {missing_total} cells are blank or NA.")
                st.session_state['df_sample'] = df
                st.success(f'Loaded {len(df)} rows and {len(df.columns)} columns')
            except Exception as e:
                st.error(readable_exception(e))

    with col2:
        # Reset button to clear dataset and table (always visible on upload page)
        def reset_selections():
            # Remove only relevant keys to avoid full Streamlit rerun issues
            for k in list(ss.keys()):
                if k.startswith('uploaded_df') or k.startswith('df_sample') or k in [
                    'features', 'target', 'step', 'model_type', 'settings', 'trained_model', 'metrics', '_X_val', '_y_val', '_y_proba', 'training_logs', 'training_status', 'training_columns', 'feature_importances', 'coefficients', 'clustering_X_scaled', 'optimal_n_clusters', 'cluster_labels']:
                    del ss[k]
            ss['has_left_upload'] = False
            init_state()
        if ss.get('uploaded_df') is not None and ss.get('has_left_upload', False):
            st.button('Reset', on_click=reset_selections, help='Clear uploaded data and start over')
        # If reset, hide selectors and data preview
        if ss.get('uploaded_df') is not None:
            df = ss['df_sample']

            # --- Feature/target selection logic (restored) ---
            def is_valid_feature_select(col):
                return not (
                    str(col).lower().startswith('auto_')
                    or str(col).lower().endswith('unique-id')
                    or str(col).lower().endswith('index')
                    or str(col) == '::auto_unique_id::'
                )
            cols = [c for c in list(ss['uploaded_df'].columns) if is_valid_feature_select(c)]
            # Filter default features to only those in current options to avoid StreamlitAPIException
            prev_features = ss.get('features')
            if prev_features is None:
                prev_features = []
            safe_defaults = [f for f in prev_features if f in cols]
            if ss['model_type'] == 'Clustering':
                features = st.multiselect('Select feature columns', options=cols, default=safe_defaults, key='feature_select')
                ss['features'] = features
                ss['target'] = None
                if features:
                    next_btn_css = """
                    <style>
                    div.stButton > button {
                        background-color: #2563eb !important;
                        color: #fff !important;
                        font-weight: bold !important;
                        border-radius: 6px !important;
                        border: none !important;
                        padding: 0.5em 2em !important;
                        font-size: 1.1rem !important;
                        box-shadow: 0 2px 8px rgba(37,99,235,0.08) !important;
                        margin-bottom: 0.5em !important;
                        transition: background 0.2s;
                    }
                    div.stButton > button:hover {
                        background-color: #1741a6 !important;
                        color: #fff !important;
                    }
                    </style>
                    """
                    st.markdown(next_btn_css, unsafe_allow_html=True)
                    if st.button('Next'):
                        ss['step'] = 2
                        st.rerun()
            else:
                def is_valid_target_select(col):
                    return not (
                        str(col).lower().startswith('auto_')
                        or str(col).lower().endswith('unique-id')
                        or str(col).lower().endswith('index')
                        or str(col) == '::auto_unique_id::'
                    )
                target_cols = [c for c in list(ss['uploaded_df'].columns) if is_valid_target_select(c)]
                target = st.selectbox('Select target column', options=["Select a target..."] + target_cols, index=0, key='target_select')
                features = st.multiselect('Select feature columns', options=cols, default=safe_defaults, key='feature_select')
                selected_target = st.session_state.get('target_select')
                if selected_target is not None and selected_target != "Select a target..." and str(selected_target).strip() != '':
                    ss['target'] = selected_target
                else:
                    ss['target'] = None
                ss['features'] = features
                if features and ss['target'] is not None and ss['target'] != "Select a target..." and str(ss['target']).strip() != '':
                    next_btn_css = """
                    <style>
                    div.stButton > button {
                        background-color: #2563eb !important;
                        color: #fff !important;
                        font-weight: bold !important;
                        border-radius: 6px !important;
                        border: none !important;
                        padding: 0.5em 2em !important;
                        font-size: 1.1rem !important;
                        box-shadow: 0 2px 8px rgba(37,99,235,0.08) !important;
                        margin-bottom: 0.5em !important;
                        transition: background 0.2s;
                    }
                    div.stButton > button:hover {
                        background-color: #1741a6 !important;
                        color: #fff !important;
                    }
                    </style>
                    """
                    st.markdown(next_btn_css, unsafe_allow_html=True)
                    st.button('Next', on_click=lambda: ss.__setitem__('step', 2))

            # --- Always show Data Preview after upload ---
            st.markdown('---')
            st.subheader('Data Preview')
            # Always show the original uploaded DataFrame, not filtered by features
            preview_df = ss['uploaded_df'] if 'uploaded_df' in ss and ss['uploaded_df'] is not None else df
            if preview_df is not None and not preview_df.empty:
                gb = GridOptionsBuilder.from_dataframe(preview_df)
                gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
                gb.configure_side_bar()
                gb.configure_grid_options(domLayout='normal')
                grid_options = gb.build()
                AgGrid(preview_df, gridOptions=grid_options, height=600, enable_enterprise_modules=False, fit_columns_on_grid_load=True)
            else:
                st.info('No data to preview.')
            st.markdown('---')

    # Removed duplicate selectors and Data Preview logic from col1
    st.markdown('---')


def step2_settings():
    st.header('2 • Training Settings')
    ss = st.session_state
    if ss['uploaded_df'] is None:
        st.info('Please upload a CSV in Step 1 first.')
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span style="background: #facc15; color: #1e293b; font-weight: bold; padding: 0.25em 0.7em; border-radius: 6px; font-size: 1.08rem;">Split the Data</span>', unsafe_allow_html=True)
        split = st.slider('Split the Data (%)', min_value=50, max_value=95, value=70, help='Choose what percent of your data to use for training (the rest is for validation).')
        ss['settings']['train_frac'] = split / 100.0
        scale = st.checkbox('Standardize numeric features', value=True)
        ss['settings']['scale'] = bool(scale)
        # Show number of training and testing rows
        if ss['uploaded_df'] is not None and ss['features'] is not None:
            df = ss['uploaded_df']
            features = ss['features']
            X = df[features].copy()
            mask = X.notna().all(axis=1)
            X = X[mask]
            n_total = len(X)
            n_train = int(n_total * ss['settings']['train_frac'])
            n_test = n_total - n_train
            st.info(f"Training rows: {n_train}, Testing rows: {n_test}")
    st.markdown('---')
    coln1, coln2 = st.columns([1, 1])
    with coln1:
        st.button('Back', on_click=lambda: ss.__setitem__('step', 1))
    with coln2:
        # Removed automatic training trigger. Training will only start from Stage 3 with a Train button.
        pass

def start_training():
    ss = st.session_state
    ss['training_status'] = 'running'
    ss['training_logs'] = []
    ss['metrics'] = None
    ss['trained_model'] = None
    ss['model_id'] = None
    ss['step'] = 3
    # Kick off training in this request (blocking but shows progress)
    # Debug: print split sizes after splitting
    try:
        df = ss['uploaded_df']
        features = ss['features']
        model_type = ss.get('model_type')
        if model_type == 'Clustering':
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            if df is not None and features is not None and len(features) > 0:
                X = df[features].copy()
                mask = X.notna().all(axis=1)
                X = X[mask]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                best_score = -1
                best_k = 2
                best_model = None
                best_labels = None
                for k in range(2, min(10, len(X_scaled))):
                    model = KMeans(n_clusters=k, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    try:
                        score = silhouette_score(X_scaled, labels)
                    except Exception:
                        score = -1
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_model = model
                        best_labels = labels
                ss['trained_model'] = best_model
                ss['cluster_labels'] = best_labels
                ss['training_status'] = 'done'
                ss['metrics'] = None
                ss['training_logs'] = [f'KMeans clustering completed. Optimal clusters: {best_k} (silhouette={best_score:.3f})']
                ss['clustering_X_scaled'] = X_scaled
                ss['clustering_scaler'] = scaler
                ss['clustering_mask'] = mask
                ss['optimal_n_clusters'] = best_k
            else:
                ss['training_status'] = 'error'
                ss['training_logs'] = ['Error: No features selected or data missing.']
        else:
            target = ss['target']
            if df is not None and features is not None and target is not None:
                X = df[features].copy()
                y = df[target].copy()
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
                train_frac = ss['settings'].get('train_frac', 0.8)
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_frac, random_state=42)
                # ...existing code...
            else:
                ss['training_status'] = 'error'
                ss['training_logs'] = ['Error: Data, features, or target missing.']
    except Exception as e:
        ss['training_status'] = 'error'
        ss['training_logs'] = [f'Error during training: {e}']


def step3_training():
    st.header('3 • Training Process')
    ss = st.session_state
    if ss['training_status'] == 'idle':
        return
    progress_placeholder = st.empty()
    p = progress_placeholder.progress(0)
    logs = []

    def log(msg):
        logs.append(msg)

    if ss['training_status'] == 'running':
        try:
            df = ss['uploaded_df'].copy()
            log('Preparing data...')
            p.progress(5)
            features = ss['features']
            target = ss['target']
            X = df[features].copy()
            y = df[target].copy()
            # Check for missing values and notify user
            missing_X = X.isna().sum().sum()
            missing_y = y.isna().sum()
            if missing_X > 0 or missing_y > 0:
                st.warning(f"Missing values detected: {missing_X} in features, {missing_y} in target. Please review your data.")
                log(f"Missing values detected: {missing_X} in features, {missing_y} in target.")
            # Do not drop or modify data, just notify
            p.progress(20)
            # handle categorical encoding: for simplicity, pd.get_dummies for categorical features
            numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            log(f'Numeric cols: {numeric_cols}; Categorical cols: {cat_cols}')
            if ss['settings'].get('scale', True) and numeric_cols:
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                ss['settings']['_scaler'] = scaler
                log('Standardized numeric features.')
            p.progress(35)
            # one-hot encode categorical
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                log('One-hot encoded categorical features.')
            p.progress(50)
            # split
            train_frac = ss['settings'].get('train_frac', 0.8)
            X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_frac, random_state=42)
            log(f'Split data: {len(X_train)} train rows, {len(X_val)} validation rows.')
            if ss['model_type'] != 'Regression':
                log(f'y_train value counts: {y_train.value_counts().to_dict()}')
                log(f'y_val value counts: {y_val.value_counts().to_dict()}')
            log(f'X_train columns: {list(X_train.columns)}')
            log(f'X_val columns: {list(X_val.columns)}')
            # Store validation data for plotting
            ss['_X_val'] = X_val
            ss['_y_val'] = y_val
            # persist training feature columns so we can align one-off/batch predictions later
            ss['training_columns'] = X_train.columns.tolist()
            log(f'Stored {len(ss["training_columns"])} training feature columns for later alignment.')
            p.progress(60)
            # choose model
            alg = ss['settings'].get('algorithm')
            model = None
            if ss['model_type'] == 'Regression':
                if alg == 'Linear Regression' or alg is None:
                    model = LinearRegression()
                    log('Fitting Linear Regression...')
            elif ss['model_type'] == 'Binary classification':
                if alg == 'Logistic Regression (Binary)' or alg is None:
                    model = LogisticRegression(C=ss['settings'].get('C', 1.0), max_iter=500, solver='lbfgs')
                    log('Fitting Logistic Regression (Binary)...')
            else:  # Multi-class classification
                if alg == 'Logistic Regression (Multi-class)' or alg is None:
                    model = LogisticRegression(C=ss['settings'].get('C', 1.0), max_iter=500, solver='lbfgs')
                    log('Fitting Logistic Regression (Multi-class)...')
            if model is None:
                log('No valid algorithm selected. Defaulting to Linear Regression.')
                model = LinearRegression()
            p.progress(70)
            # fit
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start
            log(f'Fit completed in {duration:.2f}s.')
            p.progress(85)
            # evaluate
            log('Evaluating on validation set...')
            y_pred = model.predict(X_val)
            log(f'y_pred unique values: {np.unique(y_pred)}')
            metrics = {}
            if ss['model_type'] == 'Regression':
                metrics['MAE'] = float(mean_absolute_error(y_val, y_pred))
                metrics['MSE'] = float(mean_squared_error(y_val, y_pred))
                metrics['R2'] = float(r2_score(y_val, y_pred))
                ss['metrics'] = metrics
            else:
                # for classification, if probabilities available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val)
                else:
                    y_proba = None
                metrics['accuracy'] = float(accuracy_score(y_val, y_pred))
                metrics['precision'] = float(precision_score(y_val, y_pred, average='macro', zero_division=0))
                metrics['recall'] = float(recall_score(y_val, y_pred, average='macro', zero_division=0))
                metrics['f1'] = float(f1_score(y_val, y_pred, average='macro', zero_division=0))
                ss['metrics'] = metrics
                ss['_y_val'] = y_val
                ss['_y_proba'] = y_proba
            p.progress(95)
            # store model and artifacts (in-session)
            ss['trained_model'] = model
            ss['training_status'] = 'done'
            # feature importances if applicable
            try:
                if hasattr(model, 'feature_importances_'):
                    ss['feature_importances'] = dict(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))
                elif hasattr(model, 'coef_'):
                    coefs = model.coef_.tolist()
                    # handle multiclass
                    ss['coefficients'] = coefs
            except Exception:
                pass
            p.progress(100)
            log('Training complete.')
            ss['training_logs'] = logs
            ss['step'] = 4
        except Exception as e:
            ss['training_status'] = 'error'
            st.error(readable_exception(e))
            log(f'Error during training: {str(e)}')
    # else block removed
    st.markdown('---')
    # Only show the final logs after training, not intermediate logs
    if ss.get('training_logs'):
        st.text_area('Logs', value='\n'.join(ss['training_logs']), height=200)
    if ss.get('training_status') == 'done':
        st.success('Training finished — go to Step 4 to see results.')


def step4_results():
    st.header('4 • Training Results')
    ss = st.session_state
    if ss.get('trained_model') is None:
        st.info('No trained model available. Complete Step 3 first.')
        return
    model = ss['trained_model']
    metrics = ss.get('metrics', {})
    # --- Modern card-style metrics layout ---
    st.markdown('<div style="font-weight:600;font-size:1.1rem;margin-bottom:0.7rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .metric-square {
        background: #fff;
        border-radius: 10px;
        border: 1.5px solid #e5e7eb;
        box-shadow: 0 1px 4px rgba(37,99,235,0.06);
        min-width: 120px;
        min-height: 110px;
        max-width: 160px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        padding: 0.7rem 0.5rem 0.7rem 0.5rem;
    }
    .metric-square .label {
        color: #64748b;
        font-size: 0.93rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: 0.01em;
    }
    .metric-square .value {
        color: #2563eb;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.01em;
    }
    </style>
    """, unsafe_allow_html=True)
    if ss['model_type'] == 'Clustering':
        # Clustering results: show cluster scatter plot and centroid table
        from sklearn.cluster import KMeans
        import io
        import pandas as pd
        model = ss.get('trained_model')
        X_scaled = ss.get('clustering_X_scaled')
        features = ss.get('features')
        df = ss.get('uploaded_df')
        # Always get the latest features from session state
        features = ss.get('features', [])
        if model is not None and X_scaled is not None and isinstance(features, list) and len(features) >= 2:
            # --- Clustering Model Performance Metrics ---
            inertia = getattr(model, 'inertia_', None)
            n_clusters = ss.get('optimal_n_clusters', getattr(model, 'n_clusters', None))
            labels = model.labels_ if hasattr(model, 'labels_') else ss.get('cluster_labels')
            sil_score = None
            if labels is not None and X_scaled is not None and len(set(labels)) > 1:
                from sklearn.metrics import silhouette_score
                try:
                    sil_score = silhouette_score(X_scaled, labels)
                except Exception:
                    sil_score = None
            st.markdown('<div style="font-weight:600;font-size:1.1rem;margin-bottom:0.7rem;">Model Performance</div>', unsafe_allow_html=True)
            st.markdown("""
            <style>
            .metric-square {
                background: #fff;
                border-radius: 10px;
                border: 1.5px solid #e5e7eb;
                box-shadow: 0 1px 4px rgba(37,99,235,0.06);
                min-width: 120px;
                min-height: 110px;
                max-width: 160px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
                padding: 0.7rem 0.5rem 0.7rem 0.5rem;
            }
            .metric-square .label {
                color: #64748b;
                font-size: 0.93rem;
                font-weight: 500;
                margin-bottom: 0.2rem;
                text-align: center;
                letter-spacing: 0.01em;
            }
            .metric-square .value {
                color: #2563eb;
                font-size: 1.35rem;
                font-weight: 700;
                text-align: center;
                letter-spacing: 0.01em;
            }
            </style>
            """, unsafe_allow_html=True)
            # Show clustering metrics at the top
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown(f'<div class="metric-square"><div class="label">MODEL TYPE</div><div class="value">clustering</div></div>', unsafe_allow_html=True)
            with metric_cols[1]:
                st.markdown(f'<div class="metric-square"><div class="label">CLUSTERS</div><div class="value">{n_clusters if n_clusters is not None else "-"}</div></div>', unsafe_allow_html=True)
            with metric_cols[2]:
                inertia_val = f"{inertia:.3f}" if inertia is not None else "-"
                st.markdown(f'<div class="metric-square"><div class="label">INERTIA</div><div class="value">{inertia_val}</div></div>', unsafe_allow_html=True)
            with metric_cols[3]:
                sil_val = f"{sil_score:.3f}" if sil_score is not None else "-"
                st.markdown(f'<div class="metric-square"><div class="label">SILHOUETTE</div><div class="value">{sil_val}</div></div>', unsafe_allow_html=True)
            # ...existing chart and table code...
            # Removed undefined col2 block
            centers = model.cluster_centers_ if hasattr(model, 'cluster_centers_') else None
            # Make chart 25% larger
            fig, ax = plt.subplots(figsize=(5.25, 4.0), dpi=180)
            palette = ["#2563eb", "#0ea5e9", "#facc15", "#f472b6", "#22c55e", "#eab308", "#a21caf", "#f43f5e", "#14b8a6", "#64748b"]
            for i, cluster_id in enumerate(sorted(set(labels))):
                mask_c = labels == cluster_id
                color = palette[i % len(palette)]
                ax.scatter(X_scaled[mask_c, 0], X_scaled[mask_c, 1], s=38, facecolors='none', edgecolors=color, linewidths=1.2, alpha=0.85, label=f'Cluster {cluster_id}', marker='o', zorder=2)
            # Centroids
            if centers is not None:
                ax.scatter(centers[:, 0], centers[:, 1], s=110, facecolors='none', edgecolors='#ef4444', marker='X', linewidths=2.5, label='Centroids', zorder=10)
                for idx, (cx, cy) in enumerate(centers):
                    ax.text(cx, cy, str(idx), color='#ef4444', fontsize=12, fontweight='bold', ha='center', va='center', zorder=11)
            ax.set_xlabel(features[0], fontsize=13, labelpad=2)
            ax.set_ylabel(features[1], fontsize=13, labelpad=2)
            ax.set_title('KMeans Clustering Results', fontsize=15, pad=8)
            # Move legend to right of chart
            ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
            ax.tick_params(axis='both', labelsize=11)
            ax.grid(True, linestyle=':', alpha=0.35)
            fig.tight_layout(pad=0.2)
            buf = io.BytesIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            plt.close(fig)
            svg = buf.getvalue().decode("utf-8")
            st.markdown(f"<div style='width:100%;text-align:center'>{svg}</div>", unsafe_allow_html=True)
            # Data table grouped by cluster with centroid numbers
            if df is not None and labels is not None:
                df_table = df.copy()
                df_table['Cluster'] = labels
                st.markdown('<div style="margin-top:1.2em;margin-bottom:0.3em;font-weight:600;font-size:1.08rem;">Cluster Assignments</div>', unsafe_allow_html=True)
                st.dataframe(df_table.sort_values('Cluster'), use_container_width=True)
            # Centroid coordinates table
            if centers is not None:
                centroid_df = pd.DataFrame(centers, columns=[f"{f} (scaled)" for f in features])
                centroid_df.index.name = "Centroid #"
                st.markdown('<div style="margin-top:1.2em;margin-bottom:0.3em;font-weight:600;font-size:1.08rem;">Centroid Coordinates</div>', unsafe_allow_html=True)
                st.dataframe(centroid_df, use_container_width=True)
            # --- Data Integrity & Correlation Section for Clustering ---
            # Data Integrity & Correlation table removed for clustering, as correlation with cluster labels is not meaningful.
        else:
            st.info('Not enough features to plot clusters. Select at least 2 features.')
        return
    if ss['model_type'] == 'Regression':
        cols = st.columns(7)
        def safe_metric(val, fmt=".3f"):
            try:
                if val is None:
                    return "-"
                return f"{val:{fmt}}"
            except Exception:
                return "-"
        with cols[0]:
            st.markdown('<div class="metric-square"><div class="label">MODEL TYPE</div><div class="value">regression</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="metric-square"><div class="label">TRAINING SAMPLES</div><div class="value">{len(ss["uploaded_df"])} </div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="metric-square"><div class="label">FEATURES USED</div><div class="value">{len(ss["features"])} </div></div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="metric-square"><div class="label">MEAN ABSOLUTE ERROR</div><div class="value">{safe_metric(metrics.get("MAE"))}</div></div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="metric-square"><div class="label">MEAN SQUARED ERROR</div><div class="value">{safe_metric(metrics.get("MSE"))}</div></div>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<div class="metric-square"><div class="label">ROOT MEAN SQUARED ERROR</div><div class="value">{safe_metric(metrics.get("MSE")**0.5 if metrics.get("MSE") is not None else None)}</div></div>', unsafe_allow_html=True)
        with cols[6]:
            st.markdown(f'<div class="metric-square"><div class="label">R² SCORE</div><div class="value">{safe_metric(metrics.get("R2"))}</div></div>', unsafe_allow_html=True)
    else:
        cols = st.columns(7)
        def safe_metric(val, fmt=".3f"):
            try:
                if val is None:
                    return "-"
                return f"{val:{fmt}}"
            except Exception:
                return "-"
        with cols[0]:
            st.markdown('<div class="metric-square"><div class="label">MODEL TYPE</div><div class="value">classification</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="metric-square"><div class="label">TRAINING SAMPLES</div><div class="value">{len(ss["uploaded_df"])} </div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="metric-square"><div class="label">FEATURES USED</div><div class="value">{len(ss["features"])} </div></div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="metric-square"><div class="label">ACCURACY</div><div class="value">{safe_metric(metrics.get("accuracy"))}</div></div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="metric-square"><div class="label">PRECISION (MACRO)</div><div class="value">{safe_metric(metrics.get("precision"))}</div></div>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<div class="metric-square"><div class="label">RECALL (MACRO)</div><div class="value">{safe_metric(metrics.get("recall"))}</div></div>', unsafe_allow_html=True)
        with cols[6]:
            st.markdown(f'<div class="metric-square"><div class="label">F1 (MACRO)</div><div class="value">{safe_metric(metrics.get("f1"))}</div></div>', unsafe_allow_html=True)

    # --- Keep the rest of the visuals and plots as before ---
    if ss['model_type'] == 'Regression':
        df = ss['uploaded_df']
        features = ss['features']
        target = ss['target']
        if ss.get('_y_val') is not None and ss.get('_X_val') is not None:
            y_val = ss['_y_val']
            X_val = ss['_X_val']
            try:
                y_pred = ss['trained_model'].predict(X_val)
                # Display both plots side by side
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    import io
                    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)
                    ax.scatter(
                        range(len(y_val)), y_val - y_pred,
                        alpha=0.9, s=32,
                        facecolors='none', edgecolors='#2563eb', linewidths=1.5
                    )
                    ax.axhline(0, color='k', linewidth=1.2)
                    ax.set_title('Residuals', fontsize=18)
                    ax.set_xlabel('Sample Index', fontsize=14)
                    ax.set_ylabel('Residual (y_true - y_pred)', fontsize=14)
                    ax.tick_params(axis='both', labelsize=12)
                    fig.tight_layout(pad=0.2)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="svg", bbox_inches="tight")
                    plt.close(fig)
                    svg = buf.getvalue().decode("utf-8")
                    st.markdown(f"""<div style='width:100%;text-align:center'>{svg}</div>""", unsafe_allow_html=True)
                with plot_cols[1]:
                    import io
                    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=180)
                    ax2.scatter(
                        y_val, y_pred,
                        alpha=0.9, s=32,
                        facecolors='none', edgecolors='#2563eb', linewidths=1.5
                    )
                    ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2.0)
                    ax2.set_title('True vs Predicted', fontsize=18)
                    ax2.set_xlabel('True Values', fontsize=14)
                    ax2.set_ylabel('Predicted Values', fontsize=14)
                    ax2.tick_params(axis='both', labelsize=12)
                    fig2.tight_layout(pad=0.2)
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="svg", bbox_inches="tight")
                    plt.close(fig2)
                    svg2 = buf2.getvalue().decode("utf-8")
                    st.markdown(f"""<div style='width:100%;text-align:center'>{svg2}</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not plot regression charts: {e}")
        else:
            st.info("No validation data available for plotting charts. If you see this message and have enough data, please report it.")
    elif ss['model_type'] in ('Binary classification', 'Multi-class classification'):
        # Show confusion matrix and ROC curve for binary, confusion matrix for multi-class
        if ss.get('_y_val') is not None:
            y_val = ss['_y_val']
            X_val = ss.get('_X_val')
            if X_val is not None:
                y_pred = ss['trained_model'].predict(X_val)
            else:
                y_pred = ss['trained_model'].predict(pd.get_dummies(ss['uploaded_df'][ss['features']].dropna(), drop_first=True))[:len(y_val)]
            from sklearn.metrics import ConfusionMatrixDisplay
            if ss['model_type'] == 'Binary classification' and hasattr(ss['trained_model'], 'predict_proba') and ss.get('_y_proba') is not None and len(set(y_val)) == 2:
                y_proba = ss['_y_proba']
                fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    import io
                    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)
                    disp = ConfusionMatrixDisplay.from_predictions(
                        y_val, y_pred,
                        cmap=plt.cm.Blues,
                        ax=ax,
                        colorbar=True
                    )
                    ax.set_title('Confusion Matrix', fontsize=18)
                    ax.set_xlabel('Predicted label', fontsize=14)
                    ax.set_ylabel('True label', fontsize=14)
                    ax.tick_params(axis='both', labelsize=12)
                    fig.tight_layout(pad=0.2)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="svg", bbox_inches="tight")
                    plt.close(fig)
                    svg = buf.getvalue().decode("utf-8")
                    st.markdown(f"""<div style='width:100%;text-align:center'>{svg}</div>""", unsafe_allow_html=True)
                with plot_cols[1]:
                    import io
                    fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=180)
                    ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    ax2.set_xlabel('False Positive Rate', fontsize=14)
                    ax2.set_ylabel('True Positive Rate', fontsize=14)
                    ax2.set_title('ROC Curve', fontsize=18)
                    ax2.legend(loc='lower right', fontsize=12, frameon=True)
                    ax2.tick_params(axis='both', labelsize=12)
                    fig2.tight_layout(pad=0.2)
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="svg", bbox_inches="tight")
                    plt.close(fig2)
                    svg2 = buf2.getvalue().decode("utf-8")
                    st.markdown(f"""<div style='width:100%;text-align:center'>{svg2}</div>""", unsafe_allow_html=True)
            else:
                # Multi-class or fallback confusion matrix
                class_labels = None
                if hasattr(ss['trained_model'], 'classes_'):
                    class_labels = ss['trained_model'].classes_
                import io
                fig, ax = plt.subplots(figsize=(3, 2.5), dpi=150)
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_val, y_pred,
                    display_labels=class_labels,
                    cmap=plt.cm.Blues,
                    ax=ax,
                    colorbar=True,
                    values_format='.2g' if ss['model_type'] == 'Multi-class classification' else None
                )
                # Fix tick mismatch error
                n_labels = len(class_labels) if class_labels is not None else 0
                ax.set_xticks(range(n_labels))
                ax.set_yticks(range(n_labels))
                ax.set_title('Confusion Matrix', fontsize=9 if ss['model_type'] == 'Multi-class classification' else 12, pad=5)
                ax.set_xlabel('Predicted label', fontsize=8 if ss['model_type'] == 'Multi-class classification' else 10, labelpad=4)
                ax.set_ylabel('True label', fontsize=8 if ss['model_type'] == 'Multi-class classification' else 10, labelpad=4)
                ax.tick_params(axis='both', labelsize=7 if ss['model_type'] == 'Multi-class classification' else 9, length=2)
                if ss['model_type'] == 'Multi-class classification':
                    cb = ax.figure.axes[-1]
                    cb.tick_params(labelsize=7, length=2)
                    fig.tight_layout(pad=0.5)
                else:
                    fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                st.image(buf)
    # Removed Model artifacts and download buttons as requested

    # --- Data Integrity & Correlation Section ---
    ss = st.session_state
    if ss.get('model_type') in ['Regression', 'Binary classification', 'Multi-class classification'] and ss.get('uploaded_df') is not None:
        st.markdown('---')
        st.markdown('## Data Integrity & Correlation')
        import pandas as pd
        import numpy as np
        df_corr = pd.get_dummies(ss['uploaded_df'], drop_first=False)
        selected_features = ss.get('features', [])
        target = ss.get('target')
        corr_data = []
        for feature in selected_features:
            # Numeric feature: use original column
            if feature in ss['uploaded_df'].select_dtypes(include=[np.number]).columns:
                try:
                    corr = abs(np.corrcoef(ss['uploaded_df'][feature], ss['uploaded_df'][target])[0, 1])
                except Exception:
                    corr = 0.0
            else:
                # Categorical: use max correlation of one-hot columns
                onehot_cols = [col for col in df_corr.columns if col.startswith(feature + '_')]
                if onehot_cols:
                    try:
                        corrs = [abs(np.corrcoef(df_corr[col], df_corr[target])[0, 1]) for col in onehot_cols]
                        corr = max(corrs)
                    except Exception:
                        corr = 0.0
                else:
                    corr = 0.0
            pct_impact = f"{corr*100:.1f}%"
            if corr > 0.7:
                comment = 'Strong direct impact on target.'
            elif corr > 0.3:
                comment = 'Moderate influence; may be useful.'
            elif corr < 0.1:
                comment = 'Acts as noise; little effect.'
            else:
                comment = 'Weak/uncertain effect.'
            corr_data.append({'Feature': feature, 'Max Correlation': pct_impact, 'Comment': comment})
        corr_df = pd.DataFrame(corr_data).sort_values('Max Correlation', ascending=False)
        st.markdown('### Feature Correlation & Impact Table')
        st.dataframe(corr_df[['Feature', 'Max Correlation', 'Comment']], use_container_width=True)


def step5_test():
    st.header('5 • Test Model')
    ss = st.session_state
    if ss.get('trained_model') is None:
        st.info('No trained model available.')
        return
    model = ss['trained_model']
    st.subheader('Single prediction')
    features = ss['features']
    if not features:
        st.info('No features selected.')
        return
    # generate simple inputs
    input_vals = {}
    cols = st.columns(2)
    for i, f in enumerate(features):
        dtype = ss['uploaded_df'][f].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            input_vals[f] = cols[i % 2].number_input(f, value=float(ss['uploaded_df'][f].dropna().median()))
        else:
            opts = ss['uploaded_df'][f].dropna().unique().tolist()
            input_vals[f] = cols[i % 2].selectbox(f, options=opts)
    if st.button('Predict'):
        try:
            X = pd.DataFrame([input_vals])
            # apply scaling and dummies consistent with training
            if ss['settings'].get('scale', True):
                scaler = ss['settings'].get('_scaler')
                if scaler is not None:
                    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
                    if num_cols:
                        X[num_cols] = scaler.transform(X[num_cols])
            # align dummies with training
            X = pd.get_dummies(X, columns=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c].dtype)], drop_first=True)
            # Reindex to the training columns saved during training. Fill missing columns with 0.
            trained_cols = ss.get('training_columns')
            if trained_cols is not None:
                X = X.reindex(columns=trained_cols, fill_value=0)
            pred = model.predict(X)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            else:
                proba = None
            st.success(f'Prediction: {pred[0]}')
            if proba is not None:
                st.write('Confidence / probabilities:')
                st.write(proba[0].tolist())
        except Exception as e:
            st.error(readable_exception(e))
    # Batch predictions feature removed




def main():

    init_state()
    ss = st.session_state

    # Only call the step functions and keep the bottom section

    # Inject a small JS snippet that hides floating sidebar hover controls which
    # sometimes appear as a blue rounded box. This uses heuristics on computed
    # styles (fixed/absolute position + gradient background) and a MutationObserver
    # so newly-created controls are removed immediately.
    st.components.v1.html(
        """
        <script>
        (function(){
            function hideBlueControls(){
                        try{
                            var nodes = Array.from(document.querySelectorAll('body *'));
                            nodes.forEach(function(el){
                                try{
                                    var cs = window.getComputedStyle(el);
                                    if((cs.position === 'fixed' || cs.position === 'absolute') && cs.backgroundImage && cs.backgroundImage.indexOf('gradient') !== -1){
                                        var w = el.offsetWidth || 0;
                                        var h = el.offsetHeight || 0;
                                        if(w > 16 && w < 260 && h > 16 && h < 160){
                                            el.style.setProperty('display','none','important');
                                            el.style.setProperty('visibility','hidden','important');
                                            el.style.setProperty('pointer-events','none','important');
                                            el.dataset.mllite_hidden = '1';
                                        }
                                    }
                                }catch(e){}
                            });
                        }catch(e){}
                    }
                    var obs = new MutationObserver(hideBlueControls);
                    obs.observe(document.body, { childList: true, subtree: true, attributes: true });
                    document.addEventListener('mousemove', hideBlueControls, true);
                    setTimeout(hideBlueControls, 250);
                    setInterval(hideBlueControls, 2000);
                })();
                </script>
                """,
                height=0,
        )
    sidebar_steps()
    # Help slider removed
    step = st.session_state['step']
    if step == 1:
        step1_model_and_data()
    elif step == 2:
        step2_settings()
    elif step == 3:
        # Start training automatically when entering Training step
        start_training()
        step3_training()
    elif step == 4:
        step4_results()
    elif step == 5:
        step5_test()
    # CV step and computer_vision_ui removed


if __name__ == '__main__':
    _run_with_streamlit_if_needed()