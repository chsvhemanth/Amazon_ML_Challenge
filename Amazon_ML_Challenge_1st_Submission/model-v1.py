
import os
import re
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix

# =================== Config ===================
DATASET_FOLDER = 'dataset'  # folder containing CSV files
TRAIN_CSV = os.path.join(DATASET_FOLDER, 'train.csv')
TEST_CSV  = os.path.join(DATASET_FOLDER, 'test.csv')
OUTPUT_CSV = os.path.join(DATASET_FOLDER, 'test_out.csv')
SEED = 42
NFOLD = 5           # number of CV folds
USE_GPU = True       # set False to force CPU

# =================== Helper functions ===================

def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    Useful for evaluating regression on price prediction.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom==0] = 1.0  # avoid division by zero
    return 100.0 * np.mean(np.abs(y_pred - y_true)/denom)

def clean_text(s):
    """
    Simple text cleaning:
    - replace newlines
    - strip leading/trailing spaces
    - lowercase
    - normalize multiple spaces
    """
    if pd.isna(s): 
        return ""
    s = str(s).replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
    return s.lower()

def parse_ipq(text):
    """
    Extract numeric quantity from catalog text (IPQ = Item Pack Quantity)
    Examples: "pack of 3", "2 pcs", "500 ml" -> returns numeric value
    Default fallback is 1.0
    """
    if pd.isna(text): 
        return 1.0
    s = str(text).lower()
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(?:x|pack|packs|pk|pcs|count|ct|ml|g|kg|l|litre|oz|piece|pieces)?\b', s)
    if m:
        try:
            val = float(m.group(1).replace(',', '.'))
            return val if val > 0 else 1.0
        except:
            return 1.0
    return 1.0

def make_numeric_features(df):
    """
    Generate basic numeric/text features for the model:
    - Cleaned catalog text
    - IPQ (log-transformed)
    - Length of text
    - Number of digits in text
    - Number of tokens in text
    """
    df['catalog_clean'] = df['catalog_content'].astype(str).apply(clean_text)
    df['ipq'] = df['catalog_clean'].apply(parse_ipq)
    df['ipq_log'] = np.log1p(df['ipq'])
    df['len_text'] = df['catalog_clean'].str.len().fillna(0).astype(int)
    df['num_digits'] = df['catalog_clean'].str.count(r'\d').fillna(0).astype(int)
    df['num_tokens'] = df['catalog_clean'].str.split().apply(lambda x: len(x) if isinstance(x,list) else 0).astype(int)
    return df

# =================== Main function ===================
def main():
    # Load train and test CSVs
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    # Generate numeric features
    print("Creating numeric features...")
    train = make_numeric_features(train)
    test  = make_numeric_features(test)

    # =================== Text Features ===================
    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=20000,  # reduced for speed
        ngram_range=(1,2),
        min_df=3
    )
    all_text = pd.concat([train['catalog_clean'], test['catalog_clean']], axis=0).astype(str)
    tfidf.fit(all_text)
    X_text = tfidf.transform(train['catalog_clean'])
    X_text_test = tfidf.transform(test['catalog_clean'])

    # =================== Numeric Features ===================
    dense_cols = ['ipq_log', 'len_text', 'num_digits', 'num_tokens']
    X_num = train[dense_cols].fillna(0).values
    X_num_test = test[dense_cols].fillna(0).values

    # Combine sparse (TF-IDF) and dense (numeric) features
    X_train_full = hstack([X_text, X_num]).tocsr()
    X_test_full  = hstack([X_text_test, X_num_test]).tocsr()
    y = np.log1p(train['price'].values.astype(float))

    # =================== LightGBM Parameters ===================
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': SEED
    }
    if USE_GPU:
        params['device'] = 'gpu'

    # =================== Cross-validation ===================
    oof = np.zeros(len(train))
    preds = np.zeros(X_test_full.shape[0])
    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

    print("Starting CV training...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
        print(f"\nFold {fold+1}/{NFOLD}")
        X_tr = X_train_full[tr_idx]
        X_val = X_train_full[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # Prepare LightGBM datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Early stopping & logging
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]

        # Train model
        bst = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dvalid], callbacks=callbacks)

        # Predict
        oof[val_idx] = bst.predict(X_val, num_iteration=bst.best_iteration)
        preds += bst.predict(X_test_full, num_iteration=bst.best_iteration) / NFOLD

        # Free memory
        del X_tr, X_val, dtrain, dvalid, bst
        gc.collect()

    # =================== Post-processing ===================
    oof_price = np.expm1(oof)  # back-transform from log
    pred_price = np.maximum(np.expm1(preds), 0.01)  # ensure positive prices

    # Evaluate OOF SMAPE
    oof_smape = smape(train['price'].values.astype(float), oof_price)
    print(f"OOF SMAPE: {oof_smape:.4f}%")

    # Save submission
    submission = pd.DataFrame({'sample_id': test['sample_id'], 'price': pred_price})
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission to {OUTPUT_CSV}")
    print(submission.head())

# =================== Run ===================
if __name__ == "__main__":
    main()
