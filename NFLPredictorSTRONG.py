import pandas as pd
import nfl_data_py as nfl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
pbp = nfl.import_pbp_data([2021, 2022, 2023, 2024])
print(f"Total rows loaded: {len(pbp)}")
print(f"Columns: {len(pbp.columns)}")
print(pbp['play_type'].value_counts())

# Step 2: Filter to only run and pass plays, select our features
df = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()

# Select features
features = ['down', 'ydstogo', 'yardline_100', 'score_differential',
            'game_seconds_remaining', 'shotgun', 'no_huddle',
            'posteam_type', 'posteam']

df = df[features + ['play_type']].dropna()

# Convert target to binary: pass = 1, run = 0
df['target'] = (df['play_type'] == 'pass').astype(int)

# Convert home/away to binary: home = 1, away = 0
df['home_or_away'] = (df['posteam_type'] == 'home').astype(int)

print(f"\nFiltered dataset: {len(df)} plays")
print(f"Pass: {df['target'].sum()}")
print(f"Run: {len(df) - df['target'].sum()}")
print(f"\nSample data:")
print(df.head(10))

# Step 3: Engineer team_ratio (rolling 3-game run percentage per team)
df['game_id'] = pbp.loc[df.index, 'game_id'].values

df_ratio = df[['game_id', 'posteam', 'play_type']].copy()

# Calculate run percentage per team per game
game_stats = df_ratio.groupby(['game_id', 'posteam'], as_index=False).apply(
    lambda x: pd.Series({'run_pct': (x['play_type'] == 'run').mean()}),
    include_groups=False
)

# Sort by game_id so the rolling average is chronological
game_stats = game_stats.sort_values('game_id')

# Calculate rolling 3-game average run percentage per team (shifted so we don't leak current game)
game_stats['team_ratio'] = game_stats.groupby('posteam')['run_pct'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Merge back into our main dataframe
df = df.merge(game_stats[['game_id', 'posteam', 'team_ratio']],
              on=['game_id', 'posteam'],
              how='left')

# Fill NaN team_ratio (first games of season) with league average
df['team_ratio'] = df['team_ratio'].fillna(df['team_ratio'].mean())

print(f"\nWith team_ratio added:")
print(df[['down', 'ydstogo', 'shotgun', 'team_ratio', 'target']].head(10))
print(f"Missing team_ratio values: {df['team_ratio'].isna().sum()}")

# Step 4: Prepare features and train/test split
feature_cols = ['down', 'ydstogo', 'yardline_100', 'score_differential',
                'game_seconds_remaining', 'shotgun', 'no_huddle',
                'home_or_away', 'team_ratio']

X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} plays")
print(f"Test set: {len(X_test)} plays")

# Step 5: Train models
# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"\n--- Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred, target_names=['Run', 'Pass']))

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred, target_names=['Run', 'Pass']))

# Step 6: Feature Importance (Random Forest)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='barh')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("\nFeature importances:")
for feat, imp in feat_imp.sort_values(ascending=False).items():
    print(f"  {feat}: {imp:.4f}")

# Step 7: Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, name in zip(axes, [lr_pred, rf_pred], ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Run', 'Pass'],
                yticklabels=['Run', 'Pass'], ax=ax)
    ax.set_title(f'{name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()