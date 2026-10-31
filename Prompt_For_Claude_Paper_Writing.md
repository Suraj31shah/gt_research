# Master Prompt for Writing the Complete Research Paper (IEEE Conference Format)

> **Instructions for the User:**
> 1. Copy the entire text in the box below.
> 2. Open a new chat with Claude (Claude 3.5 Sonnet / Claude 3 Opus / Claude Sonnet 4.6).
> 3. Attach or upload the following **4 Markdown files** from your project directory:
>    * `Project_Report_Context.md`
>    * `Related_Work.md`
>    * `Enron_Results.md`
>    * `Phishing_Results.md`
> 4. Paste the prompt below and run it. Claude will generate the complete, publication-grade academic paper following the exact IEEE format specifications from `Project Report Format (1).docx`.

---

```markdown
You are an expert research scientist in Cybersecurity, Game Theory, and Adversarial Machine Learning. I need you to write a complete, rigorous, publication-grade academic conference paper based on the attached empirical research project.

### FORMAT REQUIREMENTS (Strict IEEE Conference Two-Column Style from `Project Report Format (1).docx`)
1. **Title:** Capitalized like an IEEE conference title, concise, with NO subtitles, NO mathematical equations, and NO special symbols.
   - Proposed Title: "Empirical Bayesian Stackelberg Games for Adversarial Email Threat Detection: Modeling the Reality Gap and Tiered Triage"
2. **Authors & Affiliation Block:** Standard IEEE conference author block placeholder.
3. **Abstract:** 
   - Exactly ONE self-contained paragraph (180–240 words).
   - *CRITICAL IEEE GUIDELINE:* Do NOT use mathematical symbols, footnotes, references, or LaTeX equations in the abstract.
   - Must concisely summarize: the asymmetric cyber threat in email filtering, the empirical Bayesian Stackelberg formulation, the DOBSS MILP solver, the evaluation across 110,000+ real emails (Enron and Phishing corpora), the novel discovery of the "Reality Gap" (demonstrating Stackelberg's 54%–75% gap reduction over Nash), and the Budgeted Inspection tiered triage findings.
4. **Keywords:** Exactly 5 to 7 IEEE-style index terms, comma-separated (e.g., *Bayesian Stackelberg Games, Adversarial Machine Learning, Email Threat Detection, DOBSS, Reality Gap, Budgeted Inspection, Tiered Triage*).
5. **Section Headings Hierarchy:**
   - Primary Headings: Uppercase Roman Numerals (`I. INTRODUCTION`, `II. RELATED WORK`, etc.).
   - Secondary Headings: Capital letters (`A.`, `B.`, `C.`).
   - Tertiary Headings: Numbers with parentheses or italics (`1)`, `2)`).
6. **Equations:** Formally state all optimization problems and utility equations. Number every major equation consecutively in parentheses flush-right, as in `(1)`, `(2)`, `(3)`.
7. **Tables & Figures:**
   - Label all tables using Roman numerals with titles above the table: `TABLE I`, `TABLE II`, `TABLE III`, `TABLE IV`.
   - Embed COMPLETE, REAL numerical data from the attached files (NEVER use placeholders like "TBD", "N/A", or dummy numbers).
   - Reference figures as `Fig. 1`, `Fig. 2` in the text.
8. **References:** Strict IEEE citation style throughout (`[1]`, `[2]`), with a complete, publication-ready References list at the end citing the landmark papers from `Related_Work.md`.

---

### DOCUMENTATION ATTACHED AS GROUND TRUTH
I have provided four comprehensive reference files containing all technical details and experimental data:
1. `Project_Report_Context.md`: Complete mathematical definitions, utility formulas, DOBSS MILP formulation, baseline solvers, adversarial NLP engine, and cross-dataset results.
2. `Related_Work.md`: Detailed literature survey of 6 landmark papers (Dalvi 2004, Lowd & Meek 2005, Barreno 2006, Paruchuri 2008, Tambe 2011, Biggio 2013), comparison matrix, and our 5 core novelties.
3. `Enron_Results.md`: Complete empirical tables on the ~30,000 email Enron corpus (Stages 1 through 9).
4. `Phishing_Results.md`: Complete empirical tables on the 82,078 email Phishing corpus (Stages 1 through 9).

---

### REQUIRED PAPER STRUCTURE & SECTION-BY-SECTION GUIDELINES

#### I. INTRODUCTION
- Establish the operational asymmetry of modern email security: False Positives ($c_{FP}$) carry severe business disruption costs compared to False Negatives ($c_{miss}$).
- Detail the fundamental vulnerability of static NLP classifiers to test-time adversarial evasion (character perturbations, word substitutions, benign token injection).
- Identify the core limitation of previous game-theoretic security literature: reliance on arbitrary, hand-crafted payoff tables rather than real classifier performance on real text.
- Formally state our **Three Core Contributions**:
  1. *Empirically-Grounded Bayesian Stackelberg Formulation:* First game where payoff tensors are measured directly from trained NLP classifiers (NB, LR, SVM) operating on 110,000+ real emails against typed adversaries (Bulk Spammer vs. Targeted Phisher).
  2. *Quantification of the Reality Gap:* Formulation and empirical measurement of the Reality Gap ($|U_D^{\text{model}} - \mathbb{E}[U_D^{\text{realized}}]|$), proving that simultaneous Nash collapses under live deployment while Stackelberg commitment cuts the gap by 54% to 75%.
  3. *Tiered Triage via the Budgeted Inspection Game:* Formulation of computational budget constraints ($\sum x_d c_d \le B$), proving how dataset complexity dictates whether high-cost inspection (SVM) is economically rational.
- Provide a brief paragraph outlining the paper organization.

#### II. RELATED WORK & NOVELTY POSITIONING
- Group existing literature into three distinct bodies of work:
  1. *Adversarial Classification in Text:* Dalvi et al. (2004), Lowd & Meek (2005), Biggio et al. (2013). Contrast their single-model, single-attacker assumptions against our multi-model Bayesian ensemble.
  2. *Stackelberg Security Games (SSGs):* Paruchuri et al. (2008) (DOBSS) and Tambe et al. (2011) (ARMOR, PROTECT). Highlight that SSGs have historically targeted physical domains with hand-crafted payoffs; our work is the first to achieve domain transfer to cyber/email filtering with live NLP metrics.
  3. *Adversarial Machine Learning Taxonomies:* Barreno et al. (2006). Position our framework as the formal realization of their exploratory evasion defense roadmap.
- Provide a clear, formal Markdown comparison table: `TABLE I: COMPARISON OF OUR WORK WITH PRIOR LITERATURE` across Year, Game Type, Attacker Types, Payoff Source, Dataset, and Reality Gap Measurement.

#### III. GAME-THEORETIC THREAT MODEL & SYSTEM ARCHITECTURE
- Formally define the Bayesian Stackelberg Security Game:
  - Players: Defender (Leader) committing to randomized strategy $x \in \Delta(D)$, and typed Adversaries (Followers) of type $t \in \{0, 1\}$.
  - Common prior distribution $p = (p_0, p_1) = (0.70, 0.30)$.
  - Type 0 (Bulk Spammer): Low gain $G_0 = 1.0$, low budget, simple evasion toolkit $A_0 = \{0, 1, 2, 5\}$.
  - Type 1 (Targeted Phisher): High gain $G_1 = 5.0$, high sophistication, semantic-preserving toolkit $A_1 = \{0, 1, 3, 4\}$.
- Mathematical Formulation of Payoffs:
  - Write out equation for Defender Utility $U_D(t, d, a)$ with penalties $c_{miss} = 10, c_{FP} = 100, \pi_{threat} = 0.50$, and computational cost $c_{def}(d)$.
  - Write out equation for Attacker Utility $U_A(t, d, a)$.
- The DOBSS Mixed-Integer Linear Program (MILP):
  - State the full objective function and all constraints (probability simplex, attacker best-response linearization using McCormick enforcements, Big-M constraint).
- Baseline Game Solvers:
  - Zero-Sum Minimax LP formulation.
  - Simultaneous General-Sum Nash Equilibrium via Harsanyi normal-form transformation.
- The Budgeted Inspection Game (Stage 9):
  - Formulate the per-email latency/compute constraint: $\sum_{d} x_d \cdot c_{def}(d) \le \text{Budget}$.
- The Reality Gap Metric:
  - Define $\text{Reality Gap} = |U_D^{\text{model}} - \mathbb{E}[U_D^{\text{realized}}]|$, detailing the 1,000-trial Monte Carlo test simulation with 95% bootstrap confidence intervals.

#### IV. EMPIRICAL METHODOLOGY & EXPERIMENTAL SETUP
- Datasets:
  - Enron Corpus: 29,769 emails (15,791 ham, 13,978 spam).
  - Phishing Corpus: 82,078 emails (39,233 ham, 42,845 phishing).
  - Data hygiene: SHA-256 hash deduplication, stratified splits (60% Train, 15% Cal, 15% Val, 10% Test).
- Defender Ensemble (9 pure strategies):
  - Naive Bayes (Bag of Words), Logistic Regression (TF-IDF N-grams), Linear SVM (TF-IDF with Platt scaling).
  - Threshold calibration on the calibration split for three operating points: Permissive ($\text{FPR} \le 0.05$), Balanced ($\text{FPR} \le 0.01$), Strict ($\text{FPR} \le 0.001$).
- Adversarial NLP Attack Engine:
  - Black-box text-native evasion: Character Obfuscation, Synonym Replacement, Benign Ham Token Injection ("Good word attack").
  - SentenceTransformer semantic similarity gate ($\cos(e, e') \ge 0.75$) ensuring natural language deception is preserved.

#### V. EXPERIMENTAL RESULTS & REALITY GAP ANALYSIS
- `TABLE II: EQUILIBRIUM VALUES, REALIZED UTILITIES, AND REALITY GAPS (ENRON VS. PHISHING)`:
  - Include Minimax, Nash, Strong Stackelberg, and Bayesian Stackelberg (DOBSS) with Model Values, Realized Utilities (with 95% CI), Reality Gaps, and Solve Times for BOTH datasets.
- Deep Analytical Discussion of Findings:
  - Explain why Simultaneous Nash collapses upon deployment (Gap of 0.2552 on Enron, 0.2045 on Phishing).
  - Explain why Bayesian Stackelberg shrinks the Reality Gap to 0.0652 (Enron) and 0.0945 (Phishing) — leader commitment forces the attacker into an isolated best response, eliminating defender coordination failure.
  - Analyze computational scalability: DOBSS solves in $\approx 1$ second across both datasets, outperforming Nash by 32x to 65x.
- `TABLE III: DEFENDER CLASSIFIER OPERATING POINTS & CALIBRATED THRESHOLDS`:
  - Full operating table for both corpora showing thresholds, Detection Rates (DR), and False Positive Rates (FPR).
- `TABLE IV: ADVERSARIAL EVASION EFFECTIVENESS ACROSS ATTACK MODES`:
  - Full evasion rates and semantic validities, analyzing why Ham Word Injection is the dominant adversarial exploit (17.3% on Enron, 12.8% on Phishing).

#### VI. SENSITIVITY ANALYSIS & BUDGETED TIERED TRIAGE
- `TABLE V: TIERED TRIAGE UNDER COMPUTATIONAL BUDGET CONSTRAINTS (STAGE 9)`:
  - Show the rationing of SVM usage and defender utility across budgets (0.010 to 0.150).
  - Provide the deep economic insight: On Enron, SVM accuracy premium forces active budgeting (10% to 100% SVM); on Phishing, Logistic Regression achieves 96.36% DR at zero cost, strictly dominating SVM and rendering the budget constraint non-binding (0% SVM).
- Threat Prior Sensitivity:
  - Detail how equilibrium utility scales as Spammer prior shifts from 10% to 90%.
- Cost Ratio Robustness:
  - Detail how the defender strategy shifts from permissive to permanently locking into maximum detection when $c_{miss} : c_{FP}$ reaches $10:1$ and $50:1$.

#### VII. CONCLUSION & FUTURE WORK
- Summarize the paper's theoretical, algorithmic, and empirical achievements.
- Acknowledge operational limitations (e.g., stationary threat distributions, concept drift).
- Outline future avenues: Reinforcement Learning for dynamic type adaptation, Large Language Model (LLM) evasion attacks, and multi-stage triage pipelines.

#### REFERENCES
- Complete, numbered IEEE-style bibliography (`[1]`, `[2]`, `[3]`, ...) citing Paruchuri et al., Dalvi et al., Lowd & Meek, Biggio et al., Tambe et al., Barreno et al., and relevant modern NLP/adversarial learning foundations.

---

### WRITING STYLE INSTRUCTIONS
- Write with high academic rigor, precision, and formal technical vocabulary.
- Avoid vague hand-waving or informal language.
- Present all mathematical equations cleanly in LaTeX format.
- Ensure all tables are complete and formatted with GitHub Markdown tables.
- Do NOT skip sections or leave placeholders. Write the full paper text from Section I through References.
```
