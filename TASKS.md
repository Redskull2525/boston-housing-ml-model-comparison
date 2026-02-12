# Proposed Maintenance Tasks

## 1) Typo Fix Task
**Title:** Correct the malformed table-tree branch symbol in project structure

- **Issue:** In `README.md`, the project tree uses `|── Gradient_Boosting_BostonHousing.pkl` instead of `├──`, which is a formatting typo in the tree diagram.
- **Location:** `README.md` project structure block.
- **Task:** Replace `|──` with `├──` to keep tree formatting consistent.
- **Acceptance criteria:** The project structure renders with consistent tree branch symbols for sibling files.

## 2) Bug Fix Task
**Title:** Guard model selection when model loading fails

- **Issue:** `app.py` always renders `st.selectbox("Select Model", list(models.keys()))`. If model loading fails and `models` is empty, the UI can break because the select box receives no options.
- **Location:** `app.py` around model loading + model selection.
- **Task:** Add a defensive check for empty `models` and stop app execution gracefully with a clear error and recovery hint (e.g., missing `.pkl` files).
- **Acceptance criteria:** App shows a user-friendly error and does not crash when no models are loaded.

## 3) Documentation Discrepancy Task
**Title:** Fix notebook filename mismatch in README project structure

- **Issue:** README lists `Boston_Housing_Model_Comparison.ipynb`, but repository contains `Boston_Housing_Price_Prediction.ipynb`.
- **Location:** `README.md` project structure section.
- **Task:** Update README to the actual notebook filename.
- **Acceptance criteria:** The notebook filename in README exactly matches the file present in the repository.

## 4) Test Improvement Task
**Title:** Add regression test coverage for model loading and prediction path

- **Issue:** The repository has no automated tests to verify model artifact loading and prediction flow.
- **Location:** New test module (e.g., `tests/test_app_model_loading.py`).
- **Task:** Add tests that validate:
  1. All expected model artifact files are discoverable and unpickle successfully.
  2. Each loaded model returns a scalar prediction for a valid 13-feature input shape `(1, 13)`.
- **Acceptance criteria:** Tests run in CI/local with a standard test command (e.g., `pytest`) and fail clearly if artifacts are missing or incompatible.
