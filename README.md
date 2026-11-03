# ChemiLab — Cheminformatics Virtual Laboratory

A recreated, student-friendly Streamlit application based on the uploaded `app.py`.

## Features
- Home and cheminformatics theory pages
- SMILES input and validation
- 2D molecular structures using RDKit
- Interactive 3D molecular visualization using py3Dmol
- Molecular descriptors and CSV/PDF exports
- Multi-molecule comparison and chart
- Multiple-choice assessment with scoring and answer review
- Individual and combined PDF project reports

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually http://localhost:8501).

## Notes
- RDKit performs the molecular parsing and descriptor calculations.
- Interactive 3D rendering requires py3Dmol and a browser that supports its embedded viewer.
- This is a recreated implementation, not a byte-for-byte copy of the original source.
- Molecular descriptors are computed estimates and are not substitutes for experimental measurements.
