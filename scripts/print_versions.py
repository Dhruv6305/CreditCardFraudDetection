"""Prints pinned requirement lines for the packages this project uses."""
from importlib.metadata import version

for pkg in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib", "streamlit",
            "matplotlib", "shap", "kagglehub"]:
    try:
        print(f"{pkg}=={version(pkg)}")
    except Exception:
        print(f"# {pkg}: not installed")