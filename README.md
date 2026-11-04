# The Log Is Not the Link

Hands-on machine learning with public LoRaWAN measurements. Companion notebooks for the tutorial by Lorenzo Moreira Donatti and Carlos Henrique Barriquello.

## Setup

Use Python 3.12. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python scripts/fetch_artifact.py --bundle full
python scripts/acquire_public_data.py --destination .
```

The release archives supply reference results used by the notebooks. Public measurements are downloaded separately and checked against their recorded hashes. Release uploads are pending; until uploaded, use `python scripts/fetch_artifact.py --bundle full --archive /path/to/lorawan-ml-tutorial-v0.8.2-rc1.zip`.

Open the notebooks in your Jupyter-compatible editor using this environment. Start with **00 → 03 → 01 → 09**. To execute a notebook automatically:

```bash
python scripts/reproduce_notebooks.py --inprocess --pattern '03_*.ipynb'
```

## Notebooks

- [00_observations_and_prediction.ipynb](notebooks/00_observations_and_prediction.ipynb)
- [01_localization.ipynb](notebooks/01_localization.ipynb)
- [02_reception_derived_loss.ipynb](notebooks/02_reception_derived_loss.ipynb)
- [03_rssi_snr_forecasting.ipynb](notebooks/03_rssi_snr_forecasting.ipynb)
- [04_classical_and_neural_models.ipynb](notebooks/04_classical_and_neural_models.ipynb)
- [05_public_datasets.ipynb](notebooks/05_public_datasets.ipynb)
- [06_learning_curves_and_ablations.ipynb](notebooks/06_learning_curves_and_ablations.ipynb)
- [07_classification_and_calibration.ipynb](notebooks/07_classification_and_calibration.ipynb)
- [08_unsupervised_profiles.ipynb](notebooks/08_unsupervised_profiles.ipynb)
- [09_design_your_study.ipynb](notebooks/09_design_your_study.ipynb)
- [10_controlled_comparisons.ipynb](notebooks/10_controlled_comparisons.ipynb)
- [11_stability_and_reproduction.ipynb](notebooks/11_stability_and_reproduction.ipynb)

## Contents

- `notebooks/`: executable lessons, with saved outputs cleared.
- `scripts/`: Python helpers imported or invoked by the lessons and data setup.
- `data/`: source metadata, acquisition hashes and the small adaptation exercise; no downloaded measurement archives.

Paper illustrations and editorial documents are excluded. Experiments retain their original methods; these compact copies remove only external illustration displays. The complete evidence archive is a release attachment, outside the Git history.

## License and citation

Original code is distributed under the MIT license in `LICENSE`. Third-party datasets retain their source licenses. Please cite the tutorial by Lorenzo Moreira Donatti and Carlos Henrique Barriquello, *The Log Is Not the Link: A Hands-On Tutorial on Machine Learning with LoRaWAN Measurements* (working manuscript).
