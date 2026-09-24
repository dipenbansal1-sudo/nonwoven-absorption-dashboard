import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title='Nonwoven Absorption ML Dashboard', page_icon='🧪', layout='wide')

RS = 42
TEST2 = ['A5', 'B5', 'C5']
ORDER = ['1. Linear Reg', '2. Polynomial Reg', '3. SVR', '4. Random Forest', '5. XGBoost', '6. ANN (Keras)']

ROWS = [
    ('A1',0.0392,5.80,0.0097,12.91,0.065),('A2',0.0392,5.72,0.0196,6.96,0.027),
    ('A3',0.0392,6.56,0.0242,5.42,0.032),('A4',0.0392,6.61,0.0315,4.54,0.028),
    ('A5',0.0392,6.41,0.0401,4.78,0.019),('B1',0.0248,5.80,0.0097,28.55,0.078),
    ('B2',0.0248,5.56,0.0195,21.24,0.052),('B3',0.0248,5.60,0.0303,18.45,0.044),
    ('B4',0.0248,5.82,0.0388,16.04,0.035),('B5',0.0248,5.67,0.0511,14.40,0.033),
    ('C1',0.0124,5.50,0.0097,29.76,0.198),('C2',0.0124,4.83,0.0195,26.86,0.099),
    ('C3',0.0124,5.55,0.0235,21.99,0.064),('C4',0.0124,5.55,0.0303,19.35,0.079),
    ('C5',0.0124,5.45,0.0359,17.11,0.081)
]
DF = pd.DataFrame(ROWS, columns=['code','d_f_mm','thickness_mm','mu','capacity_gg','rate_ggs'])
X = DF[['d_f_mm', 'mu']]
CODES = DF['code']


def tune(est, grid, y_train):
    gs = GridSearchCV(
        est, grid, cv=KFold(5, shuffle=True, random_state=RS),
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    gs.fit(X, y_train)
    return gs.best_estimator_


def loocv_r2(est, y, use_log):
    pred = np.zeros(len(y))
    transformed = np.log(y) if use_log else y
    for tr, te in LeaveOneOut().split(X):
        model = clone(est).fit(X.iloc[tr], transformed[tr])
        value = model.predict(X.iloc[te])[0]
        pred[te] = np.exp(value) if use_log else value
    return r2_score(y, pred), pred


def test2_r2(est, y, use_log):
    mask = ~CODES.isin(TEST2)
    transformed = np.log(y) if use_log else y
    model = clone(est).fit(X[mask.values], transformed[mask.values])
    pred = model.predict(X[~mask.values])
    pred = np.exp(pred) if use_log else pred
    return r2_score(y[~mask.values], pred), pred


def build_ann(opt, lr, hidden):
    tf.keras.utils.set_random_seed(RS)
    model = keras.Sequential([
        keras.layers.Input((2,)),
        keras.layers.Dense(hidden, activation='relu'),
        keras.layers.Dense(1)
    ])
    optimizers = {
        'adam': keras.optimizers.Adam,
        'sgd': keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop
    }
    model.compile(optimizer=optimizers[opt](learning_rate=lr), loss='mse')
    return model


def ann_fit_predict(x_train, y_train, x_test, opt, lr, hidden, steps=250):
    scaler = StandardScaler().fit(x_train)
    model = build_ann(opt, lr, hidden)
    xs = scaler.transform(x_train).astype('float32')
    ys = y_train.astype('float32').reshape(-1, 1)
    for _ in range(steps):
        model.train_on_batch(xs, ys)
    return model.predict(scaler.transform(x_test).astype('float32'), verbose=0).ravel()


def ann_result(y, use_log):
    if not TF_AVAILABLE:
        return None, None, None
    xv = X.values
    transformed = np.log(y) if use_log else y
    best = None
    for opt in ['adam', 'sgd', 'rmsprop']:
        for lr in [0.01, 0.001]:
            for hidden in [8, 16]:
                p = np.zeros(len(y))
                for tr, te in KFold(5, shuffle=True, random_state=RS).split(xv):
                    pred = ann_fit_predict(xv[tr], transformed[tr], xv[te], opt, lr, hidden)
                    p[te] = np.exp(pred) if use_log else pred
                score = r2_score(y, p)
                if best is None or score > best[0]:
                    best = (score, opt, lr, hidden)
    _, opt, lr, hidden = best
    loo_p = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(xv):
        pred = ann_fit_predict(xv[tr], transformed[tr], xv[te], opt, lr, hidden)
        loo_p[te] = np.exp(pred) if use_log else pred
    mask = ~CODES.isin(TEST2)
    p2 = ann_fit_predict(xv[mask.values], transformed[mask.values], xv[~mask.values], opt, lr, hidden)
    p2 = np.exp(p2) if use_log else p2
    return (r2_score(y, loo_p), r2_score(y[~mask.values], p2)), (opt, lr, hidden), (loo_p, p2)


def model_specs():
    return {
        '1. Linear Reg': (
            Pipeline([('sc', StandardScaler()), ('m', SGDRegressor(max_iter=8000, random_state=RS))]),
            {'m__eta0':[0.001,0.01,0.05], 'm__alpha':[1e-4,1e-3,1e-2], 'm__learning_rate':['constant','invscaling']}
        ),
        '2. Polynomial Reg': (
            Pipeline([('sc', StandardScaler()), ('poly', PolynomialFeatures()), ('m', LinearRegression())]),
            {'poly__degree':[1,2,3]}
        ),
        '3. SVR': (
            Pipeline([('sc', StandardScaler()), ('m', SVR(kernel='rbf'))]),
            {'m__C':[1,10,100,300], 'm__gamma':['scale',0.1,0.5], 'm__epsilon':[0.01,0.1]}
        ),
        '4. Random Forest': (
            RandomForestRegressor(random_state=RS),
            {'n_estimators':[200,400], 'max_depth':[2,3,4,None], 'min_samples_leaf':[1,2], 'max_features':[1,2]}
        ),
        '5. XGBoost': (
            XGBRegressor(random_state=RS, verbosity=0, objective='reg:squarederror'),
            {'n_estimators':[60,150,300], 'max_depth':[2,3], 'learning_rate':[0.03,0.05,0.1], 'subsample':[0.8,1.0], 'min_child_weight':[1,2]}
        )
    }


@st.cache_data(show_spinner='Training and evaluating all models...')
def evaluate_target(target_col):
    use_log = target_col == 'rate_ggs'
    y = DF[target_col].values
    results = {}
    fitted = {}
    predictions = {}
    for name, (estimator, grid) in model_specs().items():
        transformed = np.log(y) if use_log else y
        best = tune(estimator, grid, transformed)
        loo_score, loo_pred = loocv_r2(best, y, use_log)
        test_score, test_pred = test2_r2(best, y, use_log)
        results[name] = (loo_score, test_score)
        fitted[name] = best.fit(X, transformed)
        predictions[name] = {'loocv': loo_pred, 'test2': test_pred}
    ann_scores, ann_settings, ann_preds = ann_result(y, use_log)
    if ann_scores is not None:
        results['6. ANN (Keras)'] = ann_scores
        predictions['6. ANN (Keras)'] = {'loocv': ann_preds[0], 'test2': ann_preds[1]}
    return results, fitted, predictions, ann_settings


st.title('🧪 Nonwoven Absorption ML Dashboard')
st.caption('BTP project • Raw inputs only: fibre diameter and packing density')

with st.sidebar:
    st.header('Dashboard controls')
    target_col = st.radio('Prediction target', ['capacity_gg', 'rate_ggs'], format_func=lambda x: 'Absorption capacity (g/g)' if x == 'capacity_gg' else 'Absorption rate (g/g·s)')
    selected_model = st.selectbox('Prediction model', ORDER)
    st.info('Rate is modelled in log-space, matching the original notebook.')
    if st.button('Clear cached training results'):
        st.cache_data.clear()
        st.rerun()

results, fitted, predictions, ann_settings = evaluate_target(target_col)

if '6. ANN (Keras)' not in results:
    st.warning('TensorFlow/Keras was unavailable, so the ANN model is omitted. Install tensorflow-cpu to enable it.')

# Prediction panel
st.subheader('Live prediction')
left, right = st.columns([1, 1])
with left:
    fibre_diameter = st.number_input('Fibre diameter (mm)', min_value=0.001, max_value=0.100, value=0.0248, step=0.0001, format='%.4f')
    packing_density = st.number_input('Packing density (μ)', min_value=0.001, max_value=0.100, value=0.0303, step=0.0001, format='%.4f')
    predict_clicked = st.button('Predict', type='primary', use_container_width=True)

with right:
    if selected_model == '6. ANN (Keras)' and selected_model not in fitted:
        st.error('ANN is unavailable in this environment. Select another model.')
    elif selected_model == '6. ANN (Keras)':
        st.info('The ANN model is evaluated through the notebook-style cross-validation workflow. For a production prediction endpoint, save and load a final ANN separately.')
    else:
        input_frame = pd.DataFrame([[fibre_diameter, packing_density]], columns=['d_f_mm', 'mu'])
        model = fitted[selected_model]
        raw_prediction = model.predict(input_frame)[0]
        prediction_value = float(np.exp(raw_prediction) if target_col == 'rate_ggs' else raw_prediction)
        st.metric('Predicted value', f'{prediction_value:.5f}')
        st.caption('Prediction from the selected model trained on all 15 observations.')

# Metrics
st.subheader('Evaluation metrics')
metric_cols = st.columns(3)
valid_scores = {k:v for k,v in results.items() if k in ORDER}
if valid_scores:
    best_name = max(valid_scores, key=lambda k: valid_scores[k][1])
    metric_cols[0].metric('Models evaluated', len(valid_scores))
    metric_cols[1].metric('Best A5/B5/C5 model', best_name.replace('1. ','').replace('2. ','').replace('3. ','').replace('4. ','').replace('5. ','').replace('6. ',''))
    metric_cols[2].metric('Best A5/B5/C5 R²', f'{valid_scores[best_name][1]:.3f}')

score_df = pd.DataFrame([
    {'Model': name, 'LOOCV R²': vals[0], 'A5,B5,C5 R²': vals[1]}
    for name, vals in valid_scores.items()
])
st.dataframe(score_df.style.format({'LOOCV R²':'{:.3f}', 'A5,B5,C5 R²':'{:.3f}'}), use_container_width=True, hide_index=True)

# Comparison graph
st.subheader('Model comparison')
fig, ax = plt.subplots(figsize=(11, 5.5))
plot_df = score_df.copy()
xpos = np.arange(len(plot_df))
width = 0.38
ax.bar(xpos - width/2, plot_df['LOOCV R²'], width, label='LOOCV R²')
ax.bar(xpos + width/2, plot_df['A5,B5,C5 R²'], width, label='A5,B5,C5 R²')
for i, row in plot_df.iterrows():
    ax.text(i - width/2, row['LOOCV R²'] + 0.01, f"{row['LOOCV R²']:.2f}", ha='center', fontsize=8)
    ax.text(i + width/2, row['A5,B5,C5 R²'] + 0.01, f"{row['A5,B5,C5 R²']:.2f}", ha='center', fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels([name.split('. ', 1)[1] for name in plot_df['Model']], rotation=15, ha='right')
ax.set_ylabel('R²')
ax.set_ylim(0, 1.08)
ax.set_title(f'Model comparison — target: {target_col}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

# Exploratory graphs
st.subheader('Exploratory analysis')
tab1, tab2, tab3 = st.tabs(['Dataset', 'Correlation', 'Observed relationships'])
with tab1:
    st.dataframe(DF, use_container_width=True, hide_index=True)
with tab2:
    corr = DF[['d_f_mm', 'thickness_mm', 'mu', target_col]].corr()
    st.dataframe(corr.style.format('{:.3f}'), use_container_width=True)
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    image = ax_corr.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    ax_corr.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax_corr.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center')
    fig_corr.colorbar(image, ax=ax_corr, label='Correlation')
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
with tab3:
    feature = st.selectbox('X-axis feature', ['d_f_mm', 'mu', 'thickness_mm'])
    fig_rel, ax_rel = plt.subplots(figsize=(8, 4.5))
    ax_rel.scatter(DF[feature], DF[target_col])
    for _, row in DF.iterrows():
        ax_rel.annotate(row['code'], (row[feature], row[target_col]), fontsize=8, xytext=(3,3), textcoords='offset points')
    ax_rel.set_xlabel(feature)
    ax_rel.set_ylabel(target_col)
    ax_rel.set_title(f'{target_col} vs {feature}')
    ax_rel.grid(alpha=0.3)
    fig_rel.tight_layout()
    st.pyplot(fig_rel)

st.divider()
st.caption('Methodology follows the uploaded notebook: raw inputs only, LOOCV R², and A5/B5/C5 R². Results should be interpreted cautiously because the dataset contains only 15 observations.')
