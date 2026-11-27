# DataCo Smart Supply Chain

Group repository for predictive modeling on the DataCo Smart Supply Chain dataset (V5, 180,519 rows).

## Repository Structure
- `data/DataCoSupplyChainDataset.csv` — Local raw dataset (ignored by Git; do not commit it).
- `src/preprocessing.py` — Shared target creation, leakage removal, and 80/20 train/test preprocessing.
- `notebooks/shared_preprocessing.ipynb` — Exploration and verification of the shared preprocessing implementation.
- `src/train_neural_networks.py` — PyTorch MLP neural networks for all 4 tasks (`fraud`, `late_delivery`, `sales`, `product_quantity`).
- `src/integrate_results.py` — Combines model CSVs in `results/` into `results/final_comparison_summary.csv`.
- `results/` — Evaluation CSVs and comparison charts from all team members.

## Prediction Tasks and Targets
1. **Fraud detection (classification):** `target_fraud`, derived from `Order Status == 'SUSPECTED_FRAUD'`. The original `Order Status` column is removed from model features after the target is created.
2. **Late delivery (classification):** `target_late_delivery`, derived from `Late_delivery_risk`. Delivery outcome columns are excluded from features.
3. **Sales revenue (regression):** `target_sales`, derived from `Sales`.
4. **Quantity of products ordered (regression):** `target_quantity`, derived from `Order Item Quantity`.

## Team Standards
1. **Dataset location:** Put `DataCoSupplyChainDataset.csv` inside the repository's `data/` directory. Both the notebook and Python scripts use this same location. The raw CSV is ignored by Git.
2. **Split:** 80/20 with `random_state=42`; stratify classification tasks. Imputers, encoders, and scalers are fitted on `X_train` only, then applied to `X_test`.
3. **Leakage prevention:** For fraud, derive the target from `Order Status` and then exclude `Order Status` from features. Also exclude `Delivery Status`, `Days for shipping (real)`, and `shipping date (DateOrders)`. For sales, quantity, and other regression tasks, remove the target and direct outcome-derived columns that would reveal it.
4. **Shared function:** Use `get_task_splits(df, task=...)` from `src/preprocessing.py` for every model. Valid task names are `fraud`, `late_delivery`, `sales`, and `product_quantity`.
5. **Submission:** Push changes via feature branches and Pull Requests; save evaluation metrics into `results/*.csv`.

## Quickstart
Run from the repository root:
```bash
pip install -r requirements.txt
python src/train_neural_networks.py
python src/integrate_results.py
```
