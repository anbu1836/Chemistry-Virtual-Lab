import io
import html
import streamlit as st
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors

st.set_page_config(page_title="ChemiLab | Virtual Laboratory", page_icon="🧪", layout="wide")

# -------------------- Styling --------------------
st.markdown("""
<style>
    /* Global black background and white typography */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #000000 !important;
        color: #ffffff !important;
    }
    [data-testid="stHeader"] { background: #000000 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: #080808 !important;
        color: #ffffff !important;
    }
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px;}
    h1, h2, h3, h4, h5, h6, p, label, span, li, small,
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
        color: #ffffff;
    }
    .hero {padding: 1.6rem 1.8rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg,#080808,#202020); border: 1px solid #555555;}
    .hero h1 {font-size: 2.25rem; margin: 0 0 .35rem 0; color: #ffffff;}
    .hero p {margin: 0; color: #eeeeee;}
    .small-muted {color: #dddddd; font-size: .9rem;}

    /* Inputs, dropdowns, text areas and form fields */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stDateInput input, .stTimeInput input {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        caret-color: #ffffff !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder,
    .stNumberInput input::placeholder { color: #bbbbbb !important; }
    [data-baseweb="select"] > div, [data-baseweb="popover"],
    [data-baseweb="menu"], [role="listbox"] {
        background: #000000 !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
    }
    [data-baseweb="select"] *, [role="option"] {
        color: #ffffff !important;
    }
    [role="option"]:hover, [aria-selected="true"] {
        background: #292929 !important;
    }

    /* Buttons, metrics, expanders and data containers */
    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background: #303030 !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
    }
    div[data-testid="stMetric"], [data-testid="stExpander"],
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #080808 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] details {
        background: #080808 !important;
        color: #ffffff !important;
    }
    [data-testid="stAlert"] { color: #ffffff !important; }
    [data-testid="stAlert"] p { color: #ffffff !important; }
    hr { border-color: #555555 !important; }
</style>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O"
}

def parse_smiles(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None

def bond_information(mol):
    # Add explicit H atoms so sigma bonds to hydrogen are counted.
    with_h = Chem.AddHs(Chem.Mol(mol))
    sigma = with_h.GetNumBonds()
    kekule = Chem.Mol(mol)
    pi = 0
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:
        pass
    for bond in kekule.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            pi += 1
        elif bt == Chem.BondType.TRIPLE:
            pi += 2
        elif bt == Chem.BondType.QUADRUPLE:
            pi += 3
        elif bond.GetIsAromatic():
            pi += 0.5
    return sigma, pi, with_h.GetNumAtoms()

def calculate_properties(mol):
    sigma, pi, atoms_h = bond_information(mol)
    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Sigma (σ) Bonds": sigma,
        "Pi (π) Bonds": pi,
        "Total Bond Components": sigma + pi,
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3)
    }

def make_3d(mol):
    m = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(m)
        except Exception:
            pass
    return m

def make_pdf(title, student, regno, records):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LabTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#17365D"), fontSize=19))
    story = [Paragraph(title, styles["LabTitle"]), Spacer(1, 8*mm),
             Paragraph(f"<b>Student:</b> {html.escape(student or 'Not provided')} &nbsp;&nbsp; "
                       f"<b>Registration:</b> {html.escape(regno or 'Not provided')}", styles["BodyText"]),
             Spacer(1, 6*mm)]
    for idx, (name, smiles, mol, props) in enumerate(records):
        if idx:
            story.append(PageBreak())
        story += [Paragraph(f"{idx+1}. {html.escape(name)}", styles["Heading1"]),
                  Paragraph(f"<b>Input SMILES:</b> {html.escape(smiles)}", styles["BodyText"]),
                  Paragraph(f"<b>Canonical SMILES:</b> {html.escape(Chem.MolToSmiles(mol))}", styles["BodyText"]),
                  Paragraph(f"<b>2D structure</b>", styles["Heading2"])]
        image = Draw.MolToImage(mol, size=(600, 360))
        image_buf = io.BytesIO()
        image.save(image_buf, format="PNG")
        image_buf.seek(0)
        from reportlab.platypus import Image
        story += [Image(image_buf, width=125*mm, height=75*mm), Spacer(1, 4*mm)]
        data = [["Property", "Value"]] + [[str(k), str(v)] for k, v in props.items()]
        table = Table(data, colWidths=[78*mm, 88*mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F1F5F9")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 5)
        ]))
        story.append(table)
    doc.build(story)
    return buf.getvalue()

# -------------------- Sidebar --------------------
st.sidebar.markdown("## 🧪 ChemiLab")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📖 Theory", "🧬 Molecular Visualization",
    "📊 Molecular Properties Report", "📈 Structure–Property Analysis",
    "📝 Assessment", "📄 Final Project Report"
])
st.sidebar.divider()
student_name = st.sidebar.text_input("Student name")
registration_number = st.sidebar.text_input("Registration number")
st.sidebar.caption("Learning: SMILES • 2D/3D structures • descriptors • molecular comparison")

if "smiles" not in st.session_state:
    st.session_state.smiles = EXAMPLES["Ethanol"]

def current_molecule():
    mol = parse_smiles(st.session_state.smiles)
    return mol

def show_molecule_input():
    names = list(EXAMPLES)
    selected = st.selectbox("Load an example molecule", ["Choose an example…"] + names)
    if selected != "Choose an example…" and st.button("Load selected example"):
        st.session_state.smiles = EXAMPLES[selected]
        st.rerun()
    st.text_input("SMILES notation", key="smiles", help="Example: CCO represents ethanol.")
    mol = current_molecule()
    if mol is None:
        st.error("This SMILES string is not valid. Check the spelling and try again.")
    else:
        st.success(f"Valid molecular structure • Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    return mol

# -------------------- Pages --------------------
if page == "🏠 Home":
    st.markdown("""<div class="hero"><h1>🧪 ChemiLab Virtual Laboratory</h1>
    <p>Explore molecules. Understand chemistry. Learn computational chemistry.</p></div>""",
    unsafe_allow_html=True)
    st.write("")
    st.subheader("Welcome to the Virtual Laboratory")
    st.write("An interactive learning application for cheminformatics and computational chemistry.")
    cols = st.columns(3)
    features = [("🧬", "Visualize", "Generate 2D and interactive 3D molecular structures."),
                ("📊", "Analyze", "Calculate molecular formula, weight, LogP, TPSA and more."),
                ("📄", "Report", "Compare molecules and export results for your project.")]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"### {icon} {title}")
            st.write(desc)
    st.subheader("Example molecules")
    st.dataframe(pd.DataFrame([{"Molecule": k, "SMILES": v} for k,v in EXAMPLES.items()]),
                 use_container_width=True, hide_index=True)
    st.info("Start with Molecular Visualization and load Ethanol to explore the laboratory.")

elif page == "📖 Theory":
    st.title("📖 Introduction to Cheminformatics")
    with st.expander("What is cheminformatics?", expanded=True):
        st.write("Cheminformatics combines chemistry, computer science and data analysis to store, process, visualize and analyze chemical information.")
    with st.expander("What is SMILES?", expanded=True):
        st.write("SMILES means Simplified Molecular Input Line Entry System. It represents a molecular structure using text.")
        st.table(pd.DataFrame({"Molecule":["Ethanol","Benzene","Acetic acid"],"SMILES":["CCO","c1ccccc1","CC(=O)O"]}))
    with st.expander("Molecular descriptors"):
        st.write("Descriptors are numerical values that describe a molecule. Examples include molecular weight, LogP, TPSA, hydrogen-bond donors and acceptors, and rotatable bonds.")
    with st.expander("3D molecular models"):
        st.markdown("- **Ball and stick:** atoms are spheres and bonds are sticks.\n- **Stick:** emphasizes bonds.\n- **Space filling:** shows approximate molecular volume.\n- **Wireframe:** gives a simple line representation.")

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization Laboratory")
    st.caption("Enter a SMILES string or load an example molecule.")
    mol = show_molecule_input()
    if mol is not None:
        left, right = st.columns(2)
        with left:
            st.subheader("2D molecular structure")
            st.image(Draw.MolToImage(mol, size=(700, 500)), use_container_width=True)
        with right:
            st.subheader("3D molecular structure")
            style = st.selectbox("3D display style", ["Ball and Stick", "Stick", "Space Filling", "Wireframe"])
            show_surface = st.checkbox("Show molecular surface")
            if st.button("Generate 3D model", type="primary"):
                with st.spinner("Generating 3D coordinates…"):
                    mol3d = make_3d(mol)
                if mol3d is None:
                    st.error("Could not generate a 3D model for this structure.")
                else:
                    try:
                        import py3Dmol
                        import streamlit.components.v1 as components
                        view = py3Dmol.view(width=700, height=500)
                        view.addModel(Chem.MolToMolBlock(mol3d), "mol")
                        if style == "Ball and Stick":
                            view.setStyle({"stick":{"radius":0.15},"sphere":{"scale":0.3}})
                        elif style == "Stick":
                            view.setStyle({"stick":{"radius":0.2}})
                        elif style == "Space Filling":
                            view.setStyle({"sphere":{"scale":1.0}})
                        else:
                            view.setStyle({"line":{"linewidth":2}})
                        if show_surface:
                            view.addSurface(py3Dmol.VDW, {"opacity":0.55})
                        view.setBackgroundColor("white")
                        view.zoomTo()
                        components.html(view._make_html(), height=520, scrolling=False)
                    except ImportError:
                        st.warning("Install py3Dmol to enable interactive 3D viewing. The 2D structure is available above.")

elif page == "📊 Molecular Properties Report":
    st.title("📊 Molecular Properties Report")
    mol = show_molecule_input()
    if mol is not None:
        props = calculate_properties(mol)
        st.subheader("Calculated molecular properties")
        metric_cols = st.columns(4)
        key_metrics = [("Formula", props["Molecular Formula"]), ("Molecular weight", props["Molecular Weight (g/mol)"]),
                       ("LogP", props["LogP"]), ("TPSA", props["TPSA (Å²)"])]
        for col, (label, value) in zip(metric_cols, key_metrics):
            col.metric(label, value)
        df = pd.DataFrame([{"Property":k, "Value":v} for k,v in props.items()])
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download properties (CSV)", csv, "molecular_properties.csv", "text/csv")
        pdf = make_pdf("MOLECULAR PROPERTIES REPORT", student_name, registration_number,
                       [("Molecule", st.session_state.smiles, mol, props)])
        st.download_button("⬇️ Download properties (PDF)", pdf, "molecular_properties_report.pdf", "application/pdf")

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    st.write("Compare multiple molecules. Enter one molecule per line using: **Molecule name,SMILES**")
    default = "\n".join([f"{name},{smiles}" for name,smiles in list(EXAMPLES.items())[:5]])
    text = st.text_area("Molecule list", value=default, height=190)
    rows, records, errors = [], [], []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if "," not in line:
            errors.append(f"Line {line_no}: use Name,SMILES format.")
            continue
        name, smiles = line.split(",", 1)
        mol = parse_smiles(smiles)
        if mol is None:
            errors.append(f"Line {line_no}: invalid SMILES for {name.strip()}.")
            continue
        p = calculate_properties(mol)
        rows.append({"Molecule":name.strip(), "Formula":p["Molecular Formula"],
                     "MW":p["Molecular Weight (g/mol)"], "LogP":p["LogP"],
                     "TPSA":p["TPSA (Å²)"], "HBD":p["Hydrogen Bond Donors"],
                     "HBA":p["Hydrogen Bond Acceptors"], "Rings":p["Ring Count"]})
        records.append((name.strip(), smiles.strip(), mol, p))
    for err in errors:
        st.warning(err)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download comparison (CSV)", df.to_csv(index=False).encode(),
                           "structure_property_comparison.csv", "text/csv")
        st.subheader("Compare a property")
        metric = st.selectbox("Choose property", ["MW", "LogP", "TPSA", "HBD", "HBA", "Rings"])
        st.bar_chart(df.set_index("Molecule")[[metric]])
        pdf = make_pdf("MULTI-MOLECULE ANALYSIS REPORT", student_name, registration_number, records)
        st.download_button("⬇️ Download combined PDF report", pdf, "multi_molecule_report.pdf", "application/pdf")
    else:
        st.info("Add at least one valid molecule to see a comparison.")

elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    questions = [
        ("What is the full form of SMILES?", ["Simplified Molecular Input Line Entry System","Standard Molecular Information Language","Simple Molecular Identification System"], 0),
        ("What does a SMILES string represent?", ["A molecular structure using text","A spectroscopy method","A laboratory instrument"], 0),
        ("Which SMILES represents ethanol?", ["CCO","c1ccccc1","CC(=O)O"], 0),
        ("Which SMILES represents benzene?", ["c1ccccc1","CCO","O=C=O"], 0),
        ("In SMILES, parentheses usually indicate:", ["Branches","Molecular weight","Atom colour"], 0),
        ("Which property is associated with lipophilicity?", ["LogP","TPSA","HBD"], 0),
        ("What does TPSA mean?", ["Topological Polar Surface Area","Total Pi Surface Area","Thermal Property Surface Analysis"], 0),
        ("What does HBA mean?", ["Hydrogen Bond Acceptor","Heavy Bond Atom","Hydrogen Bond Analysis"], 0),
        ("What does HBD mean?", ["Hydrogen Bond Donor","Heavy Bond Descriptor","Hydrogen Bond Distance"], 0),
        ("A double bond contains how many pi bonds?", ["1","2","0"], 0),
        ("A triple bond contains how many pi bonds?", ["2","1","3"], 0),
        ("Which model shows approximate molecular volume?", ["Space Filling","Wireframe","Text SMILES"], 0)
    ]
    with st.form("quiz_form"):
        answers = []
        for i, (question, options, correct) in enumerate(questions):
            answers.append(st.radio(f"{i+1}. {question}", options, index=None, key=f"quiz_{i}"))
        submitted = st.form_submit_button("Submit assessment", type="primary")
    if submitted:
        unanswered = sum(a is None for a in answers)
        if unanswered:
            st.warning(f"Please answer all questions. Unanswered: {unanswered}")
        else:
            score = sum(a == q[1][q[2]] for a,q in zip(answers, questions))
            st.success(f"Your score: {score} / {len(questions)} ({score/len(questions)*100:.1f}%)")
            with st.expander("Review answers"):
                for i, (q, answer) in enumerate(zip(questions, answers), 1):
                    correct = q[1][q[2]]
                    st.write(f"{i}. {q[0]}")
                    st.write(f"Your answer: {answer}")
                    st.write(f"Correct answer: {correct}")
                    st.divider()

elif page == "📄 Final Project Report":
    st.title("📄 Final Project Report")
    st.write("Generate a report for one or more molecules, including student details, 2D structures and calculated properties.")
    report_text = st.text_area("Enter molecules (one per line: Name,SMILES)",
        value="Ethanol,CCO\nBenzene,c1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O", height=150)
    records, issues = [], []
    for line_no, line in enumerate(report_text.splitlines(), 1):
        if not line.strip():
            continue
        if "," not in line:
            issues.append(f"Line {line_no}: expected Name,SMILES.")
            continue
        name, smiles = line.split(",", 1)
        mol = parse_smiles(smiles)
        if mol is None:
            issues.append(f"Line {line_no}: invalid SMILES.")
            continue
        records.append((name.strip(), smiles.strip(), mol, calculate_properties(mol)))
    for issue in issues:
        st.warning(issue)
    if records:
        st.write(f"**{len(records)} valid molecule(s) ready for the report.**")
        st.dataframe(pd.DataFrame([{"Molecule":n, "Formula":p["Molecular Formula"],
                                    "Molecular weight":p["Molecular Weight (g/mol)"]}
                                   for n,s,m,p in records]), use_container_width=True, hide_index=True)
        if st.button("Generate final PDF report", type="primary"):
            pdf = make_pdf("CHEMINFORMATICS VIRTUAL LABORATORY — FINAL PROJECT REPORT",
                           student_name, registration_number, records)
            st.download_button("⬇️ Download final project report", pdf,
                               "cheminformatics_final_project_report.pdf", "application/pdf")
    else:
        st.info("Enter at least one valid molecule.")
