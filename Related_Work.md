# Related Work & Comparison: Bayesian Stackelberg Email Threat Detection

This document maps the most closely relevant published papers to our project and compares their results directly with ours.

---

## Our Approach (Summary)

| Component | What We Did |
| :--- | :--- |
| **Framework** | Bayesian Stackelberg Game with 2 attacker types (Spammer + Phisher) solved via DOBSS (MILP) |
| **Defenders** | 3 classifiers × 3 thresholds = 9 pure defender strategies (Naive Bayes, Logistic Regression, SVM) |
| **Attacker Actions** | 6 evasion techniques: No Attack, Obfuscation, Synonym Swap (light/heavy), Ham Injection (light/heavy) |
| **Dataset** | Enron Corpus (29,767 emails: 15,791 ham + 13,976 spam/phishing) |
| **Evaluation** | Model Value vs. Realized Utility (1,000 simulation trials, 95% CI via bootstrapping) |
| **Novel Extension** | Budgeted Inspection Game (Stage 9): Tiered Triage with computational budget constraint |
| **Best Result** | Bayesian Stackelberg: Model Value = −0.6848, Realized Utility = −0.7500, Reality Gap = 0.0652 |

---

## Related Papers

---

### Paper 1 — The Foundational Basis of Our Solver

> **Paruchuri, P., Pearce, J. P., Marecki, J., Tambe, M., Ordóñez, F., & Kraus, S. (2008).**
> *"Playing Games for Security: An Efficient Exact Algorithm for Solving Bayesian Stackelberg Games."*
> **AAMAS 2008.**

#### What They Did
- Introduced **DOBSS** — the exact MILP-based algorithm used directly in our `src/games/bayesian_stackelberg.py`.
- Demonstrated it on abstract security resource allocation scenarios (scheduling guards, checkpoints).
- Application domain: Physical security (Los Angeles Airport — ARMOR system).
- **No machine learning classifiers used.** Payoff values were human-specified, not measured empirically.

#### How We Differ
| Dimension | Paruchuri et al. (2008) | **Our Work** |
| :--- | :--- | :--- |
| **Payoff Source** | Hand-crafted cost tables | **Empirically measured** from real NLP classifiers on real email text |
| **Domain** | Physical patrol scheduling | **Email / Cyber security** |
| **Attacker Model** | 1–5 abstract attacker types | **2 concrete text-attack types** (Spammer, Phisher) with 6 actual evasion strategies |
| **Evaluation** | Theoretical equilibrium value only | **Model Value + Realized Utility** on held-out email data (Reality Gap analysis) |
| **Novel Extension** | None | **Budgeted Inspection Game** (resource-constrained triage) |

#### What We Borrow
We directly implement DOBSS as the solver backbone. Our contribution is applying it to a new domain with empirically-grounded payoffs.

---

### Paper 2 — The First Game-Theoretic Formulation of Spam

> **Dalvi, N., Domingos, P., Mausam, Sanghai, S., & Verma, D. (2004).**
> *"Adversarial Classification."*
> **ACM KDD 2004.**

#### What They Did
- First paper to formally frame spam detection as a **game between a classifier and a strategic attacker**.
- Attacker is given a cost function for modifying features (e.g., adding/removing words) and solves for the minimum cost evasion path.
- Defender re-trains classifier knowing the attacker will best-respond.
- Showed their adversarially-aware classifier significantly outperforms a standard Naive Bayes filter.

#### How We Differ
| Dimension | Dalvi et al. (2004) | **Our Work** |
| :--- | :--- | :--- |
| **Game Type** | Simultaneous / Nash-style iterative | **Stackelberg (Leader-Follower)** — defender commits first |
| **Defender Model** | Single Naive Bayes classifier | **Ensemble of 9 defender strategies** (NB + LR + SVM at multiple thresholds) |
| **Attacker Types** | Single homogeneous attacker type | **Bayesian mixture of 2 types** (Spammer 70%, Phisher 30%) |
| **Payoff Estimation** | Analytical cost matrix | **Measured empirically** from real classifier performance on Enron |
| **Evaluation** | Classification accuracy under attack | **Defender utility** (accounting for asymmetric $c_{\text{miss}}$ vs. $c_{\text{FP}}$ costs) + Reality Gap |

#### Key Insight Dalvi Provided
They proved that a defender using a **static/non-strategic classifier suffers systematic exploitation**. We validate this finding empirically: our non-strategic baseline corresponds to the Zero-Sum Minimax model, which suffers the worst Reality Gap (0.2959).

---

### Paper 3 — Good Word Attacks on Text Filters

> **Lowd, D. & Meek, C. (2005).**
> *"Good Word Attacks on Statistical Spam Filters."*
> **CEAS 2005.**

#### What They Did
- Introduced the **Ham Injection / Good Word Attack** — the technique of appending common innocent words to a spam email to dilute its spam score.
- Tested against real Naive Bayes and Maximum Entropy spam filters.
- Showed that with as few as **30 words actively chosen**, an attacker could cause 50% of previously-blocked spam to reach the inbox.

#### Connection to Our Work
Our **`HamInjectionAttack`** (`strength=1` and `strength=3`) directly implements this attack. We treat it as attacker action indices 4 and 5 in our payoff tensor.

| Dimension | Lowd & Meek (2005) | **Our Work** |
| :--- | :--- | :--- |
| **Attack Studied** | Ham word injection only | **6 attack types** (obfuscation, synonyms, ham injection at multiple strengths) |
| **Evaluation** | % of spam bypassing filter | **Utility under game-theoretic equilibrium** — attacker chooses this only if it maximizes their type-specific gain |
| **Defense** | Static classifier | **Randomized mixed strategy** (probabilistic allocation across 9 defenders) |
| **Strategic Setting** | Single-stage, no formal game model | **Formal Bayesian Stackelberg game** |

#### Key Insight Lowd Provided
Ham injection is effective against Naive Bayes specifically because NB treats words independently. In our Bayesian game, Phishers (with high $c_{\text{miss}}$ gain = 5.0) rationally choose Ham Injection as their dominant attack. Bulk Spammers prefer cheaper Obfuscation (cost 0.05 vs. 0.15).

---

### Paper 4 — Adversarial Attacks on SVMs

> **Biggio, B., Corona, I., Maiorca, D., Nelson, B., et al. (2013).**
> *"Evasion Attacks against Machine Learning at Test Time."*
> **ECML PKDD 2013.**

#### What They Did
- Introduced a **gradient-based optimization framework** to craft minimal perturbations to bypass SVM and neural network classifiers.
- Adversary computes $\nabla_{\mathbf{x}} f(\mathbf{x})$ to find the gradient of the SVM decision boundary and walks the malicious sample across it.
- Applied to spam email and PDF malware detection.

#### How We Differ
| Dimension | Biggio et al. (2013) | **Our Work** |
| :--- | :--- | :--- |
| **Attack Type** | White-box gradient-based, requires model internals | **Black-box text-level attacks** (Obfuscation, Synonym Swap, Ham Injection) — semantics-preserving |
| **Semantic Check** | None (feature-space perturbations) | **Sentence Transformer similarity gate** — attacks are rejected if they are semantically incoherent |
| **Defense** | Single static SVM | **Mixed strategy across NB + LR + SVM at 3 thresholds each** |
| **Game Formulation** | None (pure attack paper) | **Full Bayesian Stackelberg framework** |
| **Defender Strategy** | Not optimized | **Computed via DOBSS** as optimal mixed defense |

#### Our Advantage
Biggio's attacks are computationally expensive and require white-box model access. Our attacker operates more realistically in a black-box setting (sends text, observes if email reached inbox), which is a more realistic threat model for email security.

---

### Paper 5 — Security Games Deployed at Real Infrastructure

> **Tambe, M. (2011).**
> *"Security and Game Theory: Algorithms, Deployed Systems, Lessons Learned."*
> **Cambridge University Press.**

(Covers ARMOR at LAX Airport, IRIS for US Air Marshals, PROTECT for US Coast Guard)

#### What They Did
- Deployed Stackelberg Security Game solutions at real national infrastructure.
- ARMOR: Randomized vehicle checkpoint scheduling at Los Angeles Airport.
- PROTECT: Optimized US Coast Guard patrol routes with budget constraints (similar to Stage 9).
- Incorporated **bounded rationality** (Quantal Response) to model the fact that human attackers make imperfect best-responses.

#### Comparison with Our Results
| Dimension | Tambe et al. (2011) | **Our Work** |
| :--- | :--- | :--- |
| **Domain** | Physical security (airports, ports) | **Cyber security (email gateway)** |
| **Attacker Model** | Human adversaries with bounded rationality | **Rational best-responding adversaries** (text-level attacks) |
| **Budget Constraint** | Yes — resource-limited patrols (PROTECT) | **Yes — Stage 9 Budgeted Inspection Game** (cheap NB/LR vs. expensive SVM) |
| **Evaluation** | Field reports & operational metrics | **Quantitative utility tables** with Reality Gap measurement |
| **Solver** | DOBSS (MILP) + ERASER + BRASS | **DOBSS (MILP)** — same family |

#### Key Validation
The PROTECT system's budget-constrained patrol optimization is the direct physical-security analog of our Stage 9 Budgeted Inspection Game. We adapt the same Stackelberg framework to the email domain with computational cost as the budget.

---

### Paper 6 — Adversarial Learning Arms Race (Survey Perspective)

> **Barreno, M., Nelson, B., Sears, R., Joseph, A. D., & Tygar, J. D. (2006).**
> *"Can Machine Learning Be Secure?"*
> **ASIACCS 2006.**

#### What They Did
- Provided the first systematic taxonomy of attacks on machine learning systems.
- Defined two key categories: **Causative Attacks** (manipulate training data, i.e. poisoning) vs. **Exploratory Attacks** (manipulate test data, i.e. evasion).
- Argued that security-critical ML systems require adversarial evaluation frameworks, not just standard accuracy benchmarks.

#### How Our Work Aligns
Our entire project is designed precisely for this adversarial evaluation framework:

| Their Taxonomy | Our Implementation |
| :--- | :--- |
| Exploratory (Evasion) Attacks | ObfuscationAttack, SynonymSwapAttack, HamInjectionAttack |
| Evaluation under adversary best-response | Stage 6: Realized Utility simulation |
| Cost-benefit analysis | $c_{\text{miss}} = 10$, $c_{\text{FP}} = 100$ in `configs/costs.yaml` |
| Robustness to parameter changes | Stage 8: Ablation on priors and cost ratios |

---

## Overall Comparison Table

| Paper | Year | Game Type | Attacker Types | Payoff Source | Dataset | Reality Gap Measured? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Dalvi et al. | 2004 | Nash (iterative) | 1 | Analytical | Synthetic | ✗ |
| Lowd & Meek | 2005 | None (attack study) | 1 | Empirical | Proprietary | ✗ |
| Barreno et al. | 2006 | Conceptual taxonomy | N/A | N/A | N/A | ✗ |
| Paruchuri et al. (DOBSS) | 2008 | Bayesian Stackelberg | Multiple | Hand-crafted | Abstract | ✗ |
| Biggio et al. | 2013 | None (attack study) | 1 | Gradient-based | Email/PDF | ✗ |
| Tambe (ARMOR/PROTECT) | 2011 | Bayesian Stackelberg | Multiple | Hand-crafted | Real (airports) | Partially |
| **Our Work** | **2024** | **Bayesian Stackelberg** | **2 (typed)** | **Empirically measured from NLP classifiers** | **Enron (29,767 real emails)** | **✓ (Reality Gap = 0.0652)** |

---

## Key Claims We Can Make in the Paper

> [!IMPORTANT]
> Based on this comparison, these are the **novel contributions** of our work:

1. **First to empirically measure payoffs from real NLP classifiers on real email text**, rather than relying on hand-crafted payoff tables (unlike Paruchuri 2008, Tambe 2011).
2. **First to introduce a typed Bayesian attacker model for email security** — distinguishing between low-sophistication bulk Spammers (70%) and high-value Phishers (30%) in a single Bayesian Stackelberg game.
3. **First to measure the Reality Gap** (Model Value vs. Realized Utility under live simulation) for email security games — proving Stackelberg's superiority over Nash (Gap: 0.0652 vs. 0.2552) and Minimax (Gap: 0.2959).
4. **First to introduce a Budgeted Inspection Game (Tiered Triage) for email security** — adapting PROTECT-style resource constraints to computational cost allocation across cheap (NB/LR) and expensive (SVM) classifiers.

