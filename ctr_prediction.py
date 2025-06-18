"""CTR prediction models for the Criteo dataset.

This module implements preprocessing and three models:
logistic regression, XGBoost and DeepFM.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim


def load_criteo(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load Criteo dataset assuming label in first column."""
    data = pd.read_csv(path)
    y = data.iloc[:, 0].values
    X = data.drop(data.columns[0], axis=1)
    return X, y


def train_val_test_split(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train, valid and test (80/10/10)."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ---------------------------------------------------------------------------
# Logistic Regression and XGBoost use hashed sparse features
# ---------------------------------------------------------------------------

def hash_features(df: pd.DataFrame, n_features: int = 2 ** 20) -> 'csr_matrix':
    """Convert dataframe to hashed feature matrix."""
    fh = FeatureHasher(n_features=n_features, input_type='string')
    return fh.transform(df.astype(str).values)


def train_logistic_regression(X_train, y_train) -> SGDClassifier:
    clf = SGDClassifier(loss='log_loss', penalty='l2', learning_rate='constant', eta0=0.01, max_iter=1)
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# DeepFM implementation
# ---------------------------------------------------------------------------

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: List[int], embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.field_dims = field_dims
        offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.register_buffer('offset', torch.tensor(offset))
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.offset.unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims: List[int]):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), 1)
        offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.longlong)
        self.register_buffer('offset', torch.tensor(offset))
        nn.init.zeros_(self.fc.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.offset.unsqueeze(0)
        return torch.sum(self.fc(x), dim=1)


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = 0.5 * (square_of_sum - sum_of_square)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return ix


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        for dim in dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepFM(nn.Module):
    def __init__(self, field_dims: List[int], embed_dim: int = 10, mlp_dims: Tuple[int, ...] = (128, 64, 32), dropout: float = 0.2):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, list(mlp_dims), dropout)
        self.output = nn.Linear(mlp_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed_x = self.embedding(x)
        linear_part = self.linear(x)
        fm_part = self.fm(embed_x)
        mlp_in = embed_x.view(-1, self.embed_output_dim)
        mlp_out = self.output(self.mlp(mlp_in))
        out = linear_part + fm_part + mlp_out
        return torch.sigmoid(out.squeeze(1))


def preprocess_for_deepfm(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Encode categorical columns as integer indices for embeddings."""
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    field_dims = []
    mappings = {}
    for col in cat_cols:
        values = train_df[col].astype('category').cat.categories
        mapping = {v: i for i, v in enumerate(values)}
        mappings[col] = mapping
        field_dims.append(len(mapping) + 1)  # +1 for unknown
    def convert(df):
        arrs = []
        for col in cat_cols:
            mapping = mappings[col]
            arr = df[col].map(mapping).fillna(len(mapping)).astype(int).values
            arrs.append(arr)
        return np.stack(arrs, axis=1)
    train_idx = convert(train_df)
    test_idx = convert(test_df)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long), field_dims


def train_deepfm(train_x: torch.Tensor, train_y: np.ndarray, valid_x: torch.Tensor, valid_y: np.ndarray, field_dims: List[int]) -> DeepFM:
    model = DeepFM(field_dims, embed_dim=10, mlp_dims=(128, 64, 32), dropout=0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_y_tensor = torch.tensor(train_y, dtype=torch.float32).to(device)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32).to(device)

    train_x = train_x.to(device)
    valid_x = valid_x.to(device)

    best_loss = float('inf')
    for epoch in range(3):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(valid_x)
            val_loss = criterion(val_pred, valid_y_tensor)
        if val_loss < best_loss:
            best_loss = val_loss
        else:
            break
    return model


def evaluate_torch_model(model: nn.Module, X_test: torch.Tensor, y_test: np.ndarray) -> Tuple[float, float]:
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
    auc = roc_auc_score(y_test, preds)
    loss = log_loss(y_test, preds)
    return auc, loss



def evaluate_sklearn_model(model, X_test, y_test):
    """Evaluate scikit-learn style model."""
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    loss = log_loss(y_test, preds)
    return auc, loss

