from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)

titanic_pipe = Pipeline([
    # scale
    ('scaler', StandardScaler()),
    ('Logit', LogisticRegression(C=0.0005, random_state=42)),
])
