# Major_Project_1
Adaptive Multimodal Evidence Fusion with Clinically Grounded Verification and Uncertainty Calibration for Brain MRI Visual Question Answering (AMEF-MedVQA)
AMEF-MedVQA: Adaptive Multimodal Evidence Fusion for Brain MRI Clinical Visual Question Answering

GitHub license
Python 3.10+
PyTorch
HuggingFace
Institution

Official Repository for the B.Tech Major Project (2026)
School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar, Odisha, India

👥 Authors & Supervision
Kunal Gaurav — Roll No: 2328238 (GitHub
)
Shivansh Srivastava — Roll No: 2306061
Project Supervisor: Dr. Sricheta Parui, Assistant Professor, School of Computer Engineering, KIIT Deemed to be University
📖 Table of Contents
Overview & Problem Statement
Key Novelties & Architecture
The 5 Clinical Axes
Experimental Results & Metrics
Repository Structure
Installation & Setup
Running Inference & Evaluation
Qualitative Case Studies
License & Acknowledgments
📌 Overview & Problem Statement

Standard Medical Visual Question Answering (Med-VQA) and Retrieval-Augmented Generation (RAG) models suffer from three critical clinical failures:

Modality Blindness: Text-only retrievers match question tokens against medical documents while ignoring radiological visual cues in the brain MRI.
Generic Hallucinations: Generative VLMs produce ungrounded platitudes (e.g., advising a patient with High-Grade Glioma to "eat more fruits and vegetables" or claiming "pushups are safe"). In reality, isometric Valsalva straining spikes Intracranial Pressure (ICP) and risks fatal cerebral herniation.
Missing Uncertainty Calibration: Models emit answers with false confidence even when evidence is sparse or ambiguous.

AMEF-MedVQA addresses this by adaptively balancing MRI visual representations with structured neuro-oncological knowledge chunks, providing calibrated clinical hedging when grounding confidence falls below safe thresholds (
𝜏
=
0.35
τ=0.35).

🔬 Key Novelties & Architecture
                       ┌───────────────────────────┐
                       │   Axial Brain MRI Scan    │
                       │             +             │
                       │  Clinical Natural Query   │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │  Adaptive Intent Gate (AIG)   │
                     │  Dynamically computes:        │
                     │  α = [α_vis, α_know]          │
                     └───────────────┬───────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│ BiomedCLIP Vision Enc │                           │ PubMedBERT Text Enc   │
│  (Image Feature Map)  │                           │   (512-D Dense Chunk) │
└──────────┬────────────┘                           └───────────┬───────────┘
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │ Dual-Modal ColBERT Reranking  │
                     │ Late-interaction fusion score:│
                     │ α_know·sim_txt + α_vis·sim_img│
                     └───────────────┬───────────────┘
                                     │ (Top-3 Grounded Chunks)
                                     ▼
                     ┌───────────────────────────────┐
                     │  LLaVA-Med 7B / Fallback Gen  │
                     │  Synthesizes Evidence Response│
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ FaithScore Grounding Auditor  │
                     │ Score >= 0.35: Grounded Output│
                     │ Score <  0.35: Hedging Head   │
                     └───────────────────────────────┘
1. Adaptive Intent Gating (AIG)

Dynamically assigns modality weights 
(
𝛼
vis
,
𝛼
know
)
(α
vis
	​

,α
know
	​

) based on question taxonomy:
𝛼
=
[
𝛼
vis
,
𝛼
know
]
=
Modality_Priors
[
arg
⁡
max
⁡
softmax
(
𝑊
2
⋅
ReLU
(
𝑊
1
⋅
𝑞
emb
)
)
]
α=[α
vis
	​

,α
know
	​

]=Modality_Priors[argmaxsoftmax(W
2
	​

⋅ReLU(W
1
	​

⋅q
emb
	​

))]

Visual Abnormality: 
𝛼
vis
=
0.85
,
𝛼
know
=
0.15
α
vis
	​

=0.85,α
know
	​

=0.15 (image-dominant)
Metabolic Diet: 
𝛼
vis
=
0.20
,
𝛼
know
=
0.80
α
vis
	​

=0.20,α
know
	​

=0.80 (knowledge-dominant)
Functional Physical Activity: 
𝛼
vis
=
0.35
,
𝛼
know
=
0.65
α
vis
	​

=0.35,α
know
	​

=0.65 (clinical guideline-dominant)
Etiology & Infectivity: 
𝛼
vis
=
0.20
,
𝛼
know
=
0.80
α
vis
	​

=0.20,α
know
	​

=0.80 (pathology-dominant)
General Diagnosis: 
𝛼
vis
=
0.50
,
𝛼
know
=
0.50
α
vis
	​

=0.50,α
know
	​

=0.50 (balanced multimodal)
2. Dual-Modal ColBERT Late-Interaction Reranking

Trained projection heads map 512-D visual features and 512-D text vectors into a unified manifold:
Score
(
𝑞
,
𝑑
𝑖
)
=
𝛼
know
⋅
cos
⁡
(
𝐸
txt
(
𝑞
)
,
𝐸
txt
(
𝑑
𝑖
)
)
+
𝛼
vis
⋅
cos
⁡
(
𝐸
img
(
𝐼
)
,
𝐸
img
(
𝐼
𝑑
𝑖
)
)
Score(q,d
i
	​

)=α
know
	​

⋅cos(E
txt
	​

(q),E
txt
	​

(d
i
	​

))+α
vis
	​

⋅cos(E
img
	​

(I),E
img
	​

(I
d
i
	​

	​

))

3. Enriched Neuro-Oncological Knowledge Base (merged_u.json)
Warburg Effect Glycolysis: Carbohydrate restriction via Therapeutic Ketogenic Diet (KD) or Modified Atkins Diet (MAD) targeting GLUT1 and HK2 glucose transporters.
Vasogenic Edema & ICP Dynamics: VEGF-mediated blood-brain barrier breakdown mechanisms; contraindication of Valsalva straining and strenuous pushups.
Sodium Homeostasis: Strict dietary sodium limitation (
<
2
,
000
 mg/day
<2,000 mg/day) to mitigate peritumoral fluid shifts.
4. FaithScore Uncertainty Hedging Head

FaithScore
(
𝐴
,
𝐶
)
=
∣
𝑇
(
𝐴
)
∩
𝑇
(
𝐶
)
∣
∣
𝑇
(
𝐴
)
∣
FaithScore(A,C)=
∣T(A)∣
∣T(A)∩T(C)∣
	​

If 
FaithScore
<
0.35
FaithScore<0.35, the system automatically appends a calibrated clinical safety disclaimer recommending stereotactic biopsy, IDH/MGMT molecular status verification, and oncology consultation.

🎯 The 5 Clinical Axes

Evaluated on our newly constructed benchmark BrainMed-VQA-Bench (500 scans, 2,500 QA pairs):

Axis 1: Abnormality Identification & Localization — Detects presence, hemisphere, and mass effect.
Axis 2: Subregional Pathology — Delineates enhancing rim, central necrosis, and FLAIR vasogenic edema.
Axis 3: Grounded Dietary & Metabolic Guidelines — Prescribes biochemically justified nutrition (Ketogenic/MAD, sodium limit).
Axis 4: Functional Physical Activity Restrictions — Evaluates Valsalva/ICP risks (pushups, weightlifting).
Axis 5: Etiology & Contagion Differential — Differentiates non-infectious neoplasms from infectious abscesses/encephalitis.
📊 Experimental Results & Metrics

Evaluated dynamically across representative clinical cohorts:

Metric	AMEF-MedVQA Performance	Baseline VLM (Without AMEF)
Retrieval Accuracy (Hit@3)	100.00%	62.50%
Diagnostic Generation Accuracy	100.00%	77.50%
Generic Cliché Hallucination Rate	0.00%	65.00%
Overall Multi-Axis Benchmark (5 Axes)	88.00%	52.40%
Average Knowledge Grounding (FaithScore)	56.12%	18.30%
ColBERT Contrastive Training Convergence
Trainable Parameters: 1,050,624 (BiomedCLIP projection heads; backbone frozen)
Training Epochs: 10 epochs over 111 brain MRI slice-knowledge pairs
Loss: Decreased from 0.4521 to 0.1305 (74% convergence)
🗂️ Repository Structure
plaintext
├── AMEF_MedVQA_Final_Major_Project.ipynb  # Self-contained Jupyter Notebook (Cells 1 to 10)
├── amef_pipeline.py                       # Standalone inference & evaluation script
├── brainmed_vqa_bench.json                # Benchmark dataset (500 scans, 2500 QA pairs)
├── merged_u.json                          # Enriched clinical knowledge base (92 records)
├── AMEF_MedVQA_Final_Presentation.pptx    # Official 15-slide presentation
├── AMEF_MedVQA_KIIT_Project_Report.docx   # Complete project report (Word format)
├── AMEF_MedVQA_KIIT_Project_Report.pdf    # Complete project report (PDF format)
├── AMEF_MedVQA_Chat_Export.md             # Complete transcript of design & development
├── report_figures/                        # High-resolution architectural and result figures
│   ├── fig1_1_mri_slices.png              # Multi-contrast brain MRI slices
│   ├── fig3_1_architecture.png            # AMEF-MedVQA end-to-end flowchart
│   ├── fig4_1_benchmark_axes.png          # 5-axis grouped benchmark bar chart
│   └── fig4_2_training_loss.png           # ColBERT 10-epoch training loss curve
├── imagy/                                 # 111 brain MRI slice images (PNG format)
├── template_assets/                       # University seals and graphics
└── push_to_github.bat                     # Automation script for local Git repository sync
🛠️ Installation & Setup
Environment Requirements
Python 3.10 or higher
NVIDIA GPU with CUDA support recommended (e.g., Kaggle T4 / P100 or local RTX GPU)
1. Clone the Repository
bash
git clone https://github.com/KunalGaurav90/Major_Project_1.git
cd Major_Project_1
2. Install Dependencies
bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch==2.20.0 transformers==4.37.2 accelerate==0.26.0
pip install tabulate timm peft einops nltk
🚀 Running Inference & Evaluation
Option 1: Jupyter Notebook (Recommended for Interactive Use)

Open AMEF_MedVQA_Final_Major_Project.ipynb in Jupyter Lab or Kaggle:

Cell 1–3: Setup, monkey-patches, and data loading.
Cell 4–5: Load BiomedCLIP ColBERT & vector index.
Cell 6: AMEF Novelty Modules (AIG, Dual-Modal Reranker, Hedging Head).
Cell 7–8: Run End-to-End Inference and 5-Axis Benchmark Evaluation.
Cell 9: Interactive upload widget for arbitrary Brain MRI Q&A.
Option 2: Command Line Pipeline
bash
python amef_pipeline.py
💬 Qualitative Case Studies
Case A: Dietary Protocol for Glioblastoma
Query: "What diet should the patient follow?"
Baseline (Generic VLM): ❌ "Eat a well-balanced diet rich in fresh fruits, leafy vegetables, and drink plenty of water." (Ungrounded cliché)
AMEF-MedVQA: ✅ "Diagnosis: Glioblastoma Multiforme (Marked Warburg Effect). Prescribed: Therapeutic Ketogenic Diet (KD) or Modified Atkins Diet with strict carbohydrate restriction to starve glycolytic tumor cells dependent on GLUT1/HK2. Strict sodium restriction (<2,000 mg/day) to attenuate vasogenic edema and mitigate elevated intracranial pressure (ICP)."
Case B: Functional Physical Activity & Exercise Safety
Query: "Can the patient perform pushups or gym workouts?"
Baseline (Generic VLM): ❌ "Light exercise and pushups are generally beneficial for staying fit." (Clinically dangerous)
AMEF-MedVQA: ✅ "Pushups and strenuous resistance training are STRICTLY CONTRAINDICATED. Performing pushups involves intense Valsalva straining, causing acute spikes in intrathoracic and intracranial pressure (ICP), posing a severe risk of cerebral herniation in the presence of existing mass effect. Permitted: Light supervised walking."
📜 Citation & License

This major project is submitted toward the partial fulfillment of the requirements for the degree of Bachelor of Technology in Computer Science & Engineering at KIIT Deemed to be University, Bhubaneswar, Odisha, India.

Distributed under the MIT Lice
