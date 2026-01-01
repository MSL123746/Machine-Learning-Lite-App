# Initialize Streamlit session state defaults
def init_state():
    ss = st.session_state
    if 'settings' not in ss:
        ss['settings'] = {}
    if 'step' not in ss:
        ss['step'] = 1
    if 'trained_model' not in ss:
        ss['trained_model'] = None
    if 'metrics' not in ss:
        ss['metrics'] = {}
    if 'uploaded_df' not in ss:
        ss['uploaded_df'] = None
    if 'last_mse' not in ss:
        ss['last_mse'] = None
    if 'auto_m' not in ss:
        ss['auto_m'] = None
    if 'auto_b' not in ss:
        ss['auto_b'] = None
# Imports
import streamlit as st
import numpy as np
import pandas as pd
import io
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Fallback for readable_exception if not defined
def readable_exception(e):
    return str(e)

# Fallback for computer_vision_ui if not defined
def computer_vision_ui():
    st.info('Computer Vision UI is not implemented.')
# Inject modern UI CSS for soft, rounded, minimal look (apply globally, very top)
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
        padding: 0.7rem 1.2rem 1.2rem 1.2rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        max-width: none !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    /* Reduce gap between sidebar and main content */
    .main, .block-container {
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

def step4_results():
    # --- DIAGNOSTIC DEBUGGING ---
    # (These will be filled in after X_pred is created, see below)
    # --- Restore metrics/cards row at top, styled as in image ---
    ss = st.session_state
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # Ensure y is defined from uploaded_df
    if 'uploaded_df' in ss and ss['uploaded_df'] is not None and len(ss['uploaded_df']) > 0:
        df = ss['uploaded_df']
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            y = df[num_cols[1]].values
        else:
            y = np.linspace(-5, 5, 200)
    else:
        y = np.linspace(-5, 5, 200)
    if 'trained_model' in ss and ss['trained_model'] is not None and 'features' in ss and ss['features'] is not None and 'uploaded_df' in ss and ss['uploaded_df'] is not None:
        X_pred = ss['uploaded_df'][ss['features']].copy()
        if '_scaler' in ss['settings'] and ss['settings']['_scaler'] is not None:
            scaler = ss['settings']['_scaler']
            num_cols = [c for c in X_pred.columns if pd.api.types.is_numeric_dtype(X_pred[c])]
            X_pred[num_cols] = scaler.transform(X_pred[num_cols])
        if hasattr(ss['trained_model'], 'feature_names_in_'):
            trained_cols = list(ss['trained_model'].feature_names_in_)
        else:
            trained_cols = list(X_pred.columns)
        X_pred = pd.get_dummies(X_pred)
        for col in trained_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[trained_cols]
        # Force fill any NaNs with 0 after column alignment
        if X_pred.isnull().values.any():
            st.error(f"NaNs found in X_pred after all processing!\nNaN count per column: {X_pred.isnull().sum().to_dict()}")
            st.write('X_pred head with NaNs:', X_pred.head())
            st.write('Trained columns:', trained_cols)
            st.write('X_pred columns:', list(X_pred.columns))
            y_pred = np.zeros_like(y)
            return  # HARD EXIT: do not call predict if NaNs remain
        y_pred = ss['trained_model'].predict(X_pred)
    else:
        y_pred = np.zeros_like(y)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    # --- Custom CSS for metrics row ---
    st.markdown("""
<style>
.metrics-row {display: flex; flex-direction: row; justify-content: flex-start; gap: 32px; margin-bottom: 0.5rem;}
.metric-square {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    padding: 18px 32p
                x 14px 32px;
    min-width: 140px;
    text-align: center;
    border: 1px solid #eee;
}
.metric-square .label {
    font-size: 0.85rem;
    color: #888;
    font-weight: 600;
    margin-bottom: 2px;
    letter-spacing: 0.5px;
}
.metric-square .value {
    font-size: 1.25rem;
    color: #2563eb;
    font-weight: 700;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:600;font-size:1.1rem;margin-bottom:0.7rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f'''
<div class="metrics-row">
  <div class="metric-square"><div class="label">MODEL TYPE</div><div class="value">regression</div></div>
  <div class="metric-square"><div class="label">TRAINING SAMPLES</div><div class="value">{len(ss["uploaded_df"])} </div></div>
  <div class="metric-square"><div class="label">FEATURES USED</div><div class="value">{len(ss["features"])} </div></div>
  <div class="metric-square"><div class="label">MEAN ABSOLUTE ERROR</div><div class="value">{mae:.3f}</div></div>
  <div class="metric-square"><div class="label">MEAN SQUARED ERROR</div><div class="value">{mse:.3f}</div></div>
  <div class="metric-square"><div class="label">ROOT MEAN SQUARED ERROR</div><div class="value">{rmse:.3f}</div></div>
  <div class="metric-square"><div class="label">R² SCORE</div><div class="value">{r2:.3f}</div></div>
</div>
''', unsafe_allow_html=True)


    import cv2
    from PIL import Image
    st.header('Computer Vision')
    st.write('Upload one or more images to analyze or train a model.')
    uploaded_files = st.file_uploader('Upload Image(s)', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        if 'process_images' not in st.session_state:
            st.session_state['process_images'] = False
        st.info(f'{len(uploaded_files)} image(s) uploaded.')
        process = st.button('Process All Images', key='process_all_btn')
        if process:
            st.session_state['process_images'] = True
            st.session_state['test_image_uploaded'] = False  # Reset test image state on new processing
        # Only show results and test uploader after processing
        if st.session_state['process_images']:
            st.subheader('Processing Results')
            num_files = len(uploaded_files)
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            st.success(f'{num_files} image(s) processed!')
            st.markdown('---')
            st.subheader('Model Performance Metrics (Example)')
            st.write('Accuracy: 0.95')
            st.write('Precision: 0.93')
            st.write('Recall: 0.92')
            st.write('F1 Score: 0.925')
            st.markdown('---')
            # Only show test uploader if a test image has not been uploaded yet
            if not st.session_state.get('test_image_uploaded', False):
                st.subheader('Test Model with New Image')
                test_file = st.file_uploader(
                    'Upload a test image to compare to the trained model',
                    type=['png', 'jpg', 'jpeg'],
                    key='test_image_upload_after_processing'
                )
                if test_file is not None:
                    # Store file in session state and immediately run prediction
                    st.session_state['test_image_uploaded'] = True
                    st.session_state['test_image_file_name'] = test_file.name
                    st.session_state['test_image_obj'] = test_file
                    # Run prediction logic
                    test_image = Image.open(test_file)
                    test_img_array = np.array(test_image)
                    test_gray = cv2.cvtColor(test_img_array, cv2.COLOR_RGB2GRAY)
                    import pickle
                    import os
                    model_path = 'model.pkl'
                    if os.path.exists(model_path):
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    else:
                        from sklearn.dummy import DummyClassifier
                        model = DummyClassifier(strategy='uniform')
                        X_fake = np.random.rand(10, test_gray.size)
                        y_fake = np.random.randint(0, 2, 10)
                        model.fit(X_fake, y_fake)
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                    test_feature = [test_gray.flatten()]
                    pred = model.predict(test_feature)[0]
                    st.session_state['test_image_prediction'] = pred
                    st.rerun()
            else:
                # Only show result and button to test another image
                pred = st.session_state.get('test_image_prediction', None)
                file_name = st.session_state.get('test_image_file_name', '')
                if pred is not None:
                    st.success(f'Test image {file_name} processed! Model prediction: {pred}')
                if st.button('Test Another Image'):
                    st.session_state['test_image_uploaded'] = False
                    st.session_state['test_image_prediction'] = None
                    st.session_state['test_image_file_name'] = ''
                    st.session_state['test_image_obj'] = None
                    st.rerun()







def step1_model_and_data():
    st.header('1 • Model Type & Data')
    ss = st.session_state
    col1, col2 = st.columns([2, 5])
    with col1:
        model_types = ['Regression', 'Binary classification', 'Multi-class classification']
        mt = st.selectbox('Model Type', model_types, index=model_types.index(ss.get('model_type', 'Regression')))
        ss['model_type'] = mt
        st.markdown('Upload a CSV file (max 10 MB). The app will infer a simple schema.')
        csv_file = st.file_uploader('Upload CSV', type=['csv'], help='CSV with header row')
        if csv_file is not None:
            try:
                data_bytes = csv_file.read()
                if len(data_bytes) > 10 * 1024 * 1024:
                    st.error('File too large (limit 10 MB).')
                    return
                df = pd.read_csv(io.BytesIO(data_bytes))
                auto_cols = [c for c in df.columns if 'auto' in str(c).lower() or 'unique_id' in str(c).lower() or '::auto_unique_id::' in str(c)]
                if auto_cols:
                    df = df.drop(columns=auto_cols)
                st.session_state['uploaded_df'] = df
                st.session_state['df_sample'] = df
                st.success(f'Loaded {len(df)} rows and {len(df.columns)} columns')
            except Exception as e:
                st.error(readable_exception(e))

    with col2:
        # If reset, hide selectors and data preview
        if ss['uploaded_df'] is not None:
            df = ss['df_sample']
            def reset_selections():
                ss.clear()
                init_state()
            # Only show selectors and preview if not reset (uploaded_df is not None)
            if ss['uploaded_df'] is not None:
                def is_valid_feature_select(col):
                    return not (
                        str(col).lower().startswith('auto_')
                        or str(col).lower().endswith('unique-id')
                        or str(col).lower().endswith('index')
                        or str(col) == '::auto_unique_id::'
                    )
                cols = [c for c in list(ss['uploaded_df'].columns) if is_valid_feature_select(c)]
                def is_valid_target_select(col):
                    return not (
                        str(col).lower().startswith('auto_')
                        or str(col).lower().endswith('unique-id')
                        or str(col).lower().endswith('index')
                        or str(col) == '::auto_unique_id::'
                    )
                target_cols = [c for c in list(ss['uploaded_df'].columns) if is_valid_target_select(c)]
                # Use stable keys for widgets so Streamlit tracks their values
                target = st.selectbox('Select target column', options=["Select a target..."] + target_cols, index=0, key='target_select')
                features = st.multiselect('Select feature columns (leave blank to use all except target)', options=cols, default=[], key='feature_select')
                # Only set target if the user has made a selection (not just defaulted to the first option)
                selected_target = st.session_state.get('target_select')
                if selected_target is not None and selected_target != "Select a target..." and str(selected_target).strip() != '':
                    ss['target'] = selected_target
                else:
                    ss['target'] = None
                ss['features'] = features
                # Show Next button only if dataset, features, and a real user-selected target (not placeholder) are present
                if ss['uploaded_df'] is not None and features and ss['target'] is not None and ss['target'] != "Select a target..." and str(ss['target']).strip() != '':
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
                def is_numeric_col(col):
                    if col not in df.columns:
                        return False
                    try:
                        pd.to_numeric(df[col].dropna())
                        return True
                    except Exception:
                        return False
                valid_features = [col for col in features if col in df.columns]
                num_cols = [col for col in valid_features if is_numeric_col(col)]
                summary_cols = list(num_cols)
                if target and is_numeric_col(target):
                    if target not in summary_cols:
                        summary_cols.append(target)
                if len(summary_cols) > 0:
                    st.subheader('Column Summaries')
                    st.dataframe(df[summary_cols].describe().T, use_container_width=True, height=200)
                    import streamlit.components.v1 as components
                    import base64
                    import io as _io
                    ncols = len(summary_cols)
                    if ncols == 1:
                        fig, axes = plt.subplots(1, 1, figsize=(4, 2.5), dpi=120)
                        axes = [axes]
                    else:
                        width = min(max(2.5 * ncols, 6), 18)
                        fig, axes = plt.subplots(1, ncols, figsize=(width, 2.5), dpi=100)
                        if ncols == 1:
                            axes = [axes]
                    for ax, col in zip(axes, summary_cols):
                        try:
                            data = pd.to_numeric(df[col].dropna())
                        except Exception:
                            data = df[col].dropna()
                        ax.hist(data, bins=15, color='#4F8DFD', alpha=0.8)
                        ax.set_title(col, fontsize=9)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    plt.tight_layout()
                    buf = _io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode()
                    components.html(f'<div style="overflow-x:auto; width:100%"><img src="data:image/png;base64,{img_b64}" style="min-width:400px; max-width:none;"/></div>', height=260)
                st.markdown('---')
                st.subheader('Data Preview')
                # Always show the full DataFrame in Data Preview
                full_df = ss['uploaded_df'] if 'uploaded_df' in ss and ss['uploaded_df'] is not None else df
                st.dataframe(full_df, use_container_width=True, height=600)
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
        st.write('Algorithm (recommended defaults)')
        if ss['model_type'] == 'Regression':
            alg_options = ['Linear Regression', 'Random Forest']
            prev_alg = ss['settings'].get('algorithm')
            alg_index = alg_options.index(prev_alg) if prev_alg in alg_options else 0
            alg = st.selectbox('Algorithm', alg_options, index=alg_index)
        elif ss['model_type'] == 'Binary classification':
            alg_options = ['Logistic Regression (Binary)', 'Random Forest']
            prev_alg = ss['settings'].get('algorithm')
            alg_index = alg_options.index(prev_alg) if prev_alg in alg_options else 0
            alg = st.selectbox('Algorithm', alg_options, index=alg_index)
        else:  # Multi-class classification
            alg_options = ['Logistic Regression (Multi-class)', 'Random Forest']
            prev_alg = ss['settings'].get('algorithm')
            alg_index = alg_options.index(prev_alg) if prev_alg in alg_options else 0
            alg = st.selectbox('Algorithm', alg_options, index=alg_index)
        ss['settings']['algorithm'] = alg
    with col2:
        st.subheader('Hyperparameters (minimal)')
        if ss['settings'].get('algorithm', '') == 'Random Forest':
            n_est = st.number_input('n_estimators', min_value=10, max_value=1000, value=100)
            max_depth = st.number_input('max_depth (0 = auto)', min_value=0, max_value=100, value=0)
            ss['settings']['n_estimators'] = int(n_est)
            ss['settings']['max_depth'] = int(max_depth) if max_depth > 0 else None
        elif ss['settings'].get('algorithm', '') == 'Logistic Regression':
            C = st.number_input('C (inverse reg strength)', min_value=0.01, max_value=10.0, value=1.0)
            penalty = st.selectbox('penalty', ['l2'], index=0)
            ss['settings']['C'] = float(C)
            ss['settings']['penalty'] = penalty
        else:
            st.write('Default parameters will be used.')
    st.markdown('---')
    coln1, coln2 = st.columns([1, 1])
    with coln1:
        st.button('Back', on_click=lambda: ss.__setitem__('step', 1))
    with coln2:
        train_btn_css = """
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
        st.markdown(train_btn_css, unsafe_allow_html=True)
        st.button('Start Training', on_click=start_training)




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
            st.info(f"DEBUG: X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}")
            if len(y_val) > 0:
                st.info(f"DEBUG: First 5 y_val: {y_val.head().tolist()}")
            else:
                st.warning("DEBUG: y_val is empty after split!")
    except Exception as e:
        st.warning(f"DEBUG: Exception during split debug: {e}")


def step3_training():
    st.header('3 • Training Process')
    ss = st.session_state
    if ss['training_status'] == 'idle' and ss['uploaded_df'] is None:
        st.info('No training scheduled. Go to Step 2 and click Start Training.')
        return
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    p = progress_placeholder.progress(0)
    logs = []

    def log(msg):
        logs.append(msg)
        log_placeholder.text_area('Logs', value='\n'.join(logs), height=180)

    if ss['training_status'] == 'running':
        try:
            df = ss['uploaded_df'].copy()
            log('Preparing data...')
            p.progress(5)
            features = ss['features']
            target = ss['target']
            X = df[features].copy()
            y = df[target].copy()
            # basic preprocessing: drop rows with NA in selected cols
            before = len(X)
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]
            after = len(X)
            log(f'Dropped {before - after} rows with missing values; {after} rows remain.')
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
                if alg == 'Linear Regression':
                    model = LinearRegression()
                    log('Fitting Linear Regression...')
                else:
                    model = RandomForestRegressor(n_estimators=ss['settings'].get('n_estimators', 100), max_depth=ss['settings'].get('max_depth', None), random_state=42)
                    log(f'Fitting RandomForestRegressor (n_estimators={ss["settings"].get("n_estimators", 100)})...')
            elif ss['model_type'] == 'Binary classification':
                if alg == 'Logistic Regression (Binary)':
                    model = LogisticRegression(C=ss['settings'].get('C', 1.0), max_iter=500, solver='lbfgs')
                    log('Fitting Logistic Regression (Binary)...')
                else:
                    model = RandomForestClassifier(n_estimators=ss['settings'].get('n_estimators', 100), max_depth=ss['settings'].get('max_depth', None), random_state=42)
                    log(f'Fitting RandomForestClassifier (n_estimators={ss["settings"].get("n_estimators", 100)})...')
            else:  # Multi-class classification
                if alg == 'Logistic Regression (Multi-class)':
                    model = LogisticRegression(C=ss['settings'].get('C', 1.0), max_iter=500, solver='lbfgs')
                    log('Fitting Logistic Regression (Multi-class)...')
                else:
                    model = RandomForestClassifier(n_estimators=ss['settings'].get('n_estimators', 100), max_depth=ss['settings'].get('max_depth', None), random_state=42)
                    log(f'Fitting RandomForestClassifier (n_estimators={ss["settings"].get("n_estimators", 100)})...')
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
    else:
        st.info('No training in progress. Start training from Step 2.')
    st.markdown('---')
    if ss.get('training_logs'):
        st.text_area('Logs', value='\n'.join(ss['training_logs']), height=200)
    if ss.get('training_status') == 'done':
        st.success('Training finished — go to Step 4 to see results.')


def step4_results():

    import numpy as np
    import pandas as pd
    ss = st.session_state

    # --- Manual Regression Trainer ---
    # Remove dark theme CSS for a clean, white background
    st.markdown('''
        <style>
        .main, .block-container, .stApp, .sidebar-content, .sidebar-panel {
            background: #fff !important;
            color: #222 !important;
        }
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
    ''', unsafe_allow_html=True)

    # --- Data Preparation ---
    if 'uploaded_df' in ss and ss['uploaded_df'] is not None and len(ss['uploaded_df']) > 0:
        df = ss['uploaded_df']
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            X = df[num_cols[0]].values
            y = df[num_cols[1]].values
        else:
            X = np.linspace(-5, 5, 200)
            y = 2.5 * X + 10 + np.random.normal(0, 5, size=X.shape)
    else:
        X = np.linspace(-5, 5, 200)
        y = 2.5 * X + 10 + np.random.normal(0, 5, size=X.shape)

    # --- Layout ---

    # --- Metrics Row at Top ---
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    # Ensure y_pred is always defined before metrics
    try:
        y_pred
    except NameError:
        # Fallback: use model or zeros if not defined
        if 'trained_model' in ss and ss['trained_model'] is not None:
            # Use the same features as during training
            if 'features' in ss and ss['features'] is not None and 'uploaded_df' in ss and ss['uploaded_df'] is not None:
                X_pred = ss['uploaded_df'][ss['features']].copy()
                # Handle categorical encoding if needed (match training)
                if '_scaler' in ss['settings'] and ss['settings']['_scaler'] is not None:
                    scaler = ss['settings']['_scaler']
                    num_cols = [c for c in X_pred.columns if pd.api.types.is_numeric_dtype(X_pred[c])]
                    X_pred[num_cols] = scaler.transform(X_pred[num_cols])
                # One-hot encode categorical columns to match training columns
                if hasattr(ss['trained_model'], 'feature_names_in_'):
                    trained_cols = list(ss['trained_model'].feature_names_in_)
                else:
                    trained_cols = list(X_pred.columns)
                X_pred = pd.get_dummies(X_pred)
                for col in trained_cols:
                    if col not in X_pred.columns:
                        X_pred[col] = 0
                        
                X_pred = X_pred[trained_cols]
                if X_pred.isnull().values.any():
                    st.error(f"NaNs found in X_pred before predict! NaN count per column: {X_pred.isnull().sum().to_dict()}")
                    st.write('X_pred head with NaNs:', X_pred.head())
                    st.write('Trained columns:', trained_cols)
                    st.write('X_pred columns:', list(X_pred.columns))
                    y_pred = np.zeros_like(y)
                    return
                y_pred = ss['trained_model'].predict(X_pred)
            else:
                # fallback for synthetic data
                X_input = X.reshape(-1, 1) if len(X.shape) == 1 else X
                if isinstance(X_input, pd.DataFrame) and X_input.isnull().values.any():
                    st.error(f"NaNs found in X_input before predict! NaN count per column: {X_input.isnull().sum().to_dict()}")
                    st.write('X_input head with NaNs:', X_input.head())
                    y_pred = np.zeros_like(y)
                    return
                y_pred = ss['trained_model'].predict(X_input)
        else:
            y_pred = np.zeros_like(y)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    st.markdown('<div class="metrics-row">'
        f'<div class="metric-square"><div class="label">R² SCORE</div><div class="value">{r2:.3f}</div></div>'
        f'<div class="metric-square"><div class="label">MAE</div><div class="value">{mae:.2f}</div></div>'
        f'<div class="metric-square"><div class="label">RMSE</div><div class="value">{rmse:.2f}</div></div>'
        '</div>', unsafe_allow_html=True)

    # --- Two Side-by-Side Charts ---
    import matplotlib.pyplot as plt
    chart1, chart2 = st.columns(2, gap="large")
    with chart1:
        # Residuals plot
        residuals = y - y_pred
        fig_res, ax_res = plt.subplots(figsize=(4, 3))
        ax_res.scatter(range(len(residuals)), residuals, color='#38bdf8', alpha=0.7, s=60, edgecolor='#222c36')
        ax_res.axhline(0, color='black', linewidth=1)
        ax_res.set_title('Residuals', fontsize=16)
        ax_res.set_xlabel('Sample Index', fontsize=13)
        ax_res.set_ylabel('Residual (y_true - y_pred)', fontsize=13)
        fig_res.tight_layout(pad=1.0)
        st.pyplot(fig_res)
    with chart2:
        # True vs Predicted plot
        fig_pred, ax_pred = plt.subplots(figsize=(4, 3))
        ax_pred.scatter(y, y_pred, color='#38bdf8', alpha=0.7, s=60, edgecolor='#222c36')
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        ax_pred.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
        ax_pred.set_title('True vs Predicted', fontsize=16)
        ax_pred.set_xlabel('True Values', fontsize=13)
        ax_pred.set_ylabel('Predicted Values', fontsize=13)
        fig_pred.tight_layout(pad=1.0)
        st.pyplot(fig_pred)


    # --- Results logic ---
    st.header('4 • Training Results')
    if ss.get('trained_model') is None:
        st.info('No trained model available. Complete Step 3 first.')
        return
    model = ss['trained_model']
    metrics = ss.get('metrics', {})

    # --- Keep the rest of the visuals and plots as before ---
    if ss['model_type'] == 'Regression':
        df = ss['uploaded_df']
        features = ss['features']
        target = ss['target']
        if ss.get('_y_val') is not None and ss.get('_X_val') is not None:
            y_val = ss['_y_val']
            X_val = ss['_X_val']
            try:
                if isinstance(X_val, pd.DataFrame) and X_val.isnull().values.any():
                    st.error(f"NaNs found in X_val before predict! NaN count per column: {X_val.isnull().sum().to_dict()}")
                    st.write('X_val head with NaNs:', X_val.head())
                    return
                y_pred = ss['trained_model'].predict(X_val)
                # Display both plots side by side
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    fig, ax = plt.subplots(figsize=(2.5, 2))
                    ax.scatter(range(len(y_val)), y_val - y_pred, alpha=0.7)
                    ax.axhline(0, color='k', linewidth=0.8)
                    ax.set_title('Residuals', fontsize=10)
                    ax.set_xlabel('Sample Index', fontsize=9)
                    ax.set_ylabel('Residual (y_true - y_pred)', fontsize=9)
                    ax.tick_params(axis='both', labelsize=8)
                    fig.tight_layout(pad=0.7)
                    st.pyplot(fig)
                with plot_cols[1]:
                    fig2, ax2 = plt.subplots(figsize=(2.5, 2))
                    ax2.scatter(y_val, y_pred, alpha=0.7)
                    ax2.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
                    ax2.set_title('True vs Predicted', fontsize=10)
                    ax2.set_xlabel('True Values', fontsize=9)
                    ax2.set_ylabel('Predicted Values', fontsize=9)
                    ax2.tick_params(axis='both', labelsize=8)
                    fig2.tight_layout(pad=0.7)
                    st.pyplot(fig2)
            except Exception as e:
                st.warning(f"Could not plot regression charts: {e}")
        else:
            st.info("No validation data available for plotting charts. If you see this message and have enough data, please report it.")
    else:
        # Visuals for classification
        if ss.get('_y_val') is not None:
            y_val = ss['_y_val']
            # Use the same X_val as in validation
            X_val = ss.get('_X_val')
            if X_val is not None:
                if isinstance(X_val, pd.DataFrame) and X_val.isnull().values.any():
                    st.error(f"NaNs found in X_val before predict! NaN count per column: {X_val.isnull().sum().to_dict()}")
                    st.write('X_val head with NaNs:', X_val.head())
                    return
                y_pred = ss['trained_model'].predict(X_val)
            else:
                X_pred = pd.get_dummies(ss['uploaded_df'][ss['features']].dropna(), drop_first=True)
                if X_pred.isnull().values.any():
                    st.error(f"NaNs found in fallback X_pred before predict! NaN count per column: {X_pred.isnull().sum().to_dict()}")
                    st.write('X_pred head with NaNs:', X_pred.head())
                    return
                y_pred = ss['trained_model'].predict(X_pred)[:len(y_val)]
            # Confusion matrix
            from sklearn.metrics import ConfusionMatrixDisplay
            # Show confusion matrix and ROC curve side by side for binary classification
            if hasattr(ss['trained_model'], 'predict_proba') and ss.get('_y_proba') is not None and len(set(y_val)) == 2:
                y_proba = ss['_y_proba']
                fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=150)
                    disp = ConfusionMatrixDisplay.from_predictions(
                        y_val, y_pred,
                        cmap=plt.cm.Blues,
                        ax=ax,
                        colorbar=True
                    )
                    ax.set_title('Confusion Matrix', fontsize=12)
                    ax.set_xlabel('Predicted label', fontsize=10)
                    ax.set_ylabel('True label', fontsize=10)
                    ax.tick_params(axis='both', labelsize=9)
                    fig.tight_layout()
                    st.pyplot(fig)
                with plot_cols[1]:
                    fig2, ax2 = plt.subplots(figsize=(3, 2.5), dpi=150)
                    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax2.set_xlim([0.0, 1.0])
                    ax2.set_ylim([0.0, 1.05])
                    ax2.set_xlabel('False Positive Rate', fontsize=10)
                    ax2.set_ylabel('True Positive Rate', fontsize=10)
                    ax2.set_title('ROC Curve', fontsize=12)
                    ax2.legend(loc='lower right', fontsize=7, frameon=True)
                    fig2.tight_layout()
                    st.pyplot(fig2)
            else:
                # Only one confusion matrix for multi-class, compact and clear
                if ss.get('model_type') == 'Multi-class classification':
                    # Try to use class names if available
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
                        values_format='.2g'
                    )
                    ax.set_title('Confusion Matrix', fontsize=9, pad=5)
                    ax.set_xlabel('Predicted label', fontsize=8, labelpad=4)
                    ax.set_ylabel('True label', fontsize=8, labelpad=4)
                    ax.tick_params(axis='both', labelsize=7, length=2)
                    cb = ax.figure.axes[-1]
                    cb.tick_params(labelsize=7, length=2)
                    fig.tight_layout(pad=0.5)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    st.image(buf)
                else:
                    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=150)
                    disp = ConfusionMatrixDisplay.from_predictions(
                        y_val, y_pred,
                        cmap=plt.cm.Blues,
                        ax=ax,
                        colorbar=True
                    )
                    ax.set_title('Confusion Matrix', fontsize=12)
                    ax.set_xlabel('Predicted label', fontsize=10)
                    ax.set_ylabel('True label', fontsize=10)
                    ax.tick_params(axis='both', labelsize=9)
                    fig.tight_layout()
                    st.pyplot(fig)
    # Removed Model artifacts and download buttons as requested


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
    st.markdown('---')
    st.subheader('Batch predictions')
    batch_file = st.file_uploader('Upload CSV for batch predictions', type=['csv'])
    if batch_file is not None:
        try:
            bdf = pd.read_csv(batch_file)
            Xb = bdf[features]
            # basic processing
            if ss['settings'].get('scale', True):
                scaler = ss['settings'].get('_scaler')
                num_cols = [c for c in Xb.columns if pd.api.types.is_numeric_dtype(Xb[c])]
                if scaler and num_cols:
                    Xb[num_cols] = scaler.transform(Xb[num_cols])
            Xb = pd.get_dummies(Xb, drop_first=True)
            # align columns with training columns if available
            trained_cols = ss.get('training_columns')
            if trained_cols is not None:
                Xb = Xb.reindex(columns=trained_cols, fill_value=0)
            preds = ss['trained_model'].predict(Xb)
            bdf['prediction'] = preds
            st.dataframe(bdf.head(20))
            csv_bytes = bdf.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions CSV', data=csv_bytes, file_name='predictions.csv')
        except Exception as e:
            st.error(readable_exception(e))




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
    # --- CLASSIC LEFT SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.markdown('<style>div[data-testid="stSidebarNav"] {margin-bottom: 2rem;}</style>', unsafe_allow_html=True)
        st.markdown('<h4 style="margin-bottom:1.2em;">Stage Navigation</h4>', unsafe_allow_html=True)
        nav_labels = [
            'Stage 1: Model & Data',
            'Stage 2: Settings',
            'Stage 3: Training',
            'Stage 4: Results',
            'Stage 5: Test',
        ]
        nav_steps = [1, 2, 3, 4, 5]
        for i, label in enumerate(nav_labels):
            btn = st.button(label, key=f'navbtn_{i}',
                on_click=lambda s=nav_steps[i]: ss.__setitem__('step', s),
                use_container_width=True)
        st.markdown('---')
        st.success('ML App Ready for Use!', icon='✅')
        st.button('CV: Computer Vision', on_click=lambda: ss.__setitem__('step', 'cv'), use_container_width=True)

    step = st.session_state['step']
    if step == 1:
        step1_model_and_data()
    elif step == 2:
        step2_settings()
    elif step == 3:
        step3_training()
    elif step == 4:
        step4_results()
    elif step == 5:
        step5_test()
    elif step == 'cv':
        computer_vision_ui()



# Always call main() at the top level for Streamlit
main()
