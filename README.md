# DataCo Smart Supply Chain — Machine Learning & Neural Networks

Group repository for predictive modeling on the DataCo Smart Supply Chain dataset (V5, 180,519 rows).

## Repository Structure
- `data/` — Local folder for `DataCoSupplyChainDataset.csv` (ignored by Git).
- `src/preprocessing.py` — Shared data cleaning, leakage removal, and 80/20 train/test split (`random_state=42`).
- `src/train_neural_networks.py` — PyTorch MLP neural networks for all 4 tasks (`fraud`, `late_delivery`, `sales`, `benefit_per_order`).
- `src/integrate_results.py` — Combines all model CSVs in `results/` into `results/final_comparison_summary.csv`.
- `results/` — Evaluation CSVs and comparison charts from all team members.

## Team Standards
1. **Dataset:** `DataCoSupplyChainDataset.csv` (V5) placed locally in `data/`. Do not commit raw CSVs.
2. **Split:** 80/20 split with `random_state=42` (stratified for classification). Fit scalers/encoders on `X_train` only.
3. **Leakage Removed:** `Order Status`, `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)`, plus direct price/quantity/discount/profit columns for `sales` and profit columns for `benefit_per_order`.
4. **Submission:** Push changes via feature branches and Pull Requests, saving evaluation metrics into `results/*.csv`.

## Quickstart
```bash
pip install -r requirements.txt
python src/train_neural_networks.py
python src/integrate_results.py