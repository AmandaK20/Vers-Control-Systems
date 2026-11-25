import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

SEED = 42

BASE_DROP_COLS = [
    # Targets & post-outcome columns
    'target_fraud', 'target_late_delivery', 'target_sales', 'target_profit',
    'Order Status', 'Delivery Status', 'Late_delivery_risk',
    'Days for shipping (real)', 'shipping date (DateOrders)',
    # Constant / 100% missing
    'Product Description', 'Product Status', 'Customer Password', 'Customer Email',
    # PII / Pure IDs / URLs / Raw timestamps
    'Customer Fname', 'Customer Lname', 'Customer Street', 'Product Image',
    'Order Item Id', 'Order Id', 'Customer Id', 'Order Customer Id',
    'order date (DateOrders)',
    # Exact duplicate columns
    'Product Category Id', 'Order Item Cardprod Id',
    'Order Item Product Price', 'Order Item Total', 'Order Profit Per Order',
    # High-cardinality geographic/text columns
    'Order Zipcode', 'Customer Zipcode', 'Order City', 'Customer City',
    'Order State', 'Customer State', 'Order Country', 'Product Name'
]

SALES_LEAKAGE_COLS = [
    'Sales', 'Sales per customer', 'Product Price', 'Order Item Quantity',
    'Order Item Discount', 'Order Item Discount Rate',
    'Benefit per order', 'Order Item Profit Ratio'
]

PROFIT_LEAKAGE_COLS = [
    'Benefit per order', 'Order Item Profit Ratio'
]


def load_and_prepare_base(filepath: str = 'data/DataCoSupplyChainDataset.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='latin1')

    # 4 Prediction Targets
    df['target_fraud'] = (df['Order Status'] == 'SUSPECTED_FRAUD').astype(int)
    df['target_late_delivery'] = df['Late_delivery_risk'].astype(int)
    df['target_sales'] = df['Sales'].astype(float)
    df['target_profit'] = df['Benefit per order'].astype(float)

    # Temporal features from order timestamp
    order_dt = pd.to_datetime(df['order date (DateOrders)'])
    df['order_month'] = order_dt.dt.month
    df['order_dayofweek'] = order_dt.dt.dayofweek
    df['order_hour'] = order_dt.dt.hour

    return df


def get_task_splits(df: pd.DataFrame, task: str, seed: int = SEED):
    """
    Returns preprocessed (X_train, X_test, y_train, y_test) using an 80/20 split
    with random_state=42 (stratified for classification tasks).
    """
    if task == 'fraud':
        y = df['target_fraud'].values
        drop_cols = BASE_DROP_COLS
        stratify = y
    elif task == 'late_delivery':
        y = df['target_late_delivery'].values
        drop_cols = BASE_DROP_COLS
        stratify = y
    elif task == 'sales':
        y = df['target_sales'].values
        drop_cols = BASE_DROP_COLS + SALES_LEAKAGE_COLS
        stratify = None
    elif task == 'benefit_per_order':
        y = df['target_profit'].values
        drop_cols = BASE_DROP_COLS + PROFIT_LEAKAGE_COLS
        stratify = None
    else:
        raise ValueError(f"Unknown task: {task}")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=stratify
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_proc = preprocessor.transform(X_test).astype(np.float32)

    return X_train_proc, X_test_proc, y_train, y_test