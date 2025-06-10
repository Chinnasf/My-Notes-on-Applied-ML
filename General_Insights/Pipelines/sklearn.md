

## Example of using `sklearn`'s pipeline to take care of leakage. 


```Python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Step 1: Split first!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Fit transformations only on training data
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # scaler is fit on X_train only
    ('clf', LogisticRegression())
])

# Step 3: Fit and evaluate
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

**IMPORTANT**: `StandardScaler` is fit on training data only. It then transforms test data **using the same mean/std from the training dataset**.

**HERE IS HOW YOU WOULD CONTAMINATE YOUR TEST SET**

```Python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leaks test set info!

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```