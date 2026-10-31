# Master Project Context: Empirical Bayesian Stackelberg Email Threat Detection

This document provides the complete, authoritative technical, mathematical, architectural, and experimental reference for the project: **"Empirical Bayesian Stackelberg Games for Adversarial Email Threat Detection: Modeling the Reality Gap and Tiered Triage."**

---

## 1. Problem Formulation & High-Level Architecture

Modern email security operates under an acute asymmetric arms race. While machine learning classifiers (Naive Bayes, Logistic Regression, Support Vector Machines) achieve high offline accuracy on static benchmarks, they suffer from two major flaws when deployed:
1. **Adversarial Evasion:** Malicious senders adapt their payloads using character obfuscation, synonym substitution, and benign token stuffing ("good word" injection).
2. **Asymmetric Operational Costs:** Blocking a legitimate business email (False Positive, $c_{FP}$) carries catastrophic cost compared to occasionally letting spam reach the junk folder (False Negative, $c_{miss}$). Standard static tuning cannot balance this dynamically.

To solve this, we model email filtering as a **Bayesian Stackelberg Security Game**:
* **Leader (Defender / Security Gateway):** Commits to a randomized mixed strategy $x \in \Delta(D)$ over an ensemble of $9$ distinct classifier-threshold configurations.
* **Followers (Adversaries):** Partitioned into two distinct Bayesian threat types $t \in \{0, 1\}$ (Bulk Spammers vs. Targeted Phishers) who observe the defender's mixed commitment and select the optimal evasion attack $a \in A_t$ that maximizes their expected utility.
* **Empirical Grounding:** Unlike prior theoretical literature (e.g., Paruchuri 2008, Tambe 2011) which used synthetic, hand-crafted payoffs, all payoff tensors ($U_D, U_A$) are measured directly from the empirical confusion matrices of trained NLP models evaluated on $110,000+$ real emails across two distinct corpora (Enron and Phishing).

```
+-----------------------------------------------------------------------------------+
|                           System Architecture Pipeline                             |
+-----------------------------------------------------------------------------------+
| 1. Raw Email Corpora (Enron: 30k, Phishing: 82k)                                  |
|    --> SHA-256 Deduplication & Cleaning                                          |
|    --> Stratified Splits (Train: 60%, Cal: 15%, Val: 15%, Test: 10%)              |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
| 2. Defender Ensemble (Stage 2)          | 3. Adversarial NLP Engine (Stage 3)     |
|    - Naive Bayes (Bag of Words)         |    - No Attack (Baseline)               |
|    - Logistic Regression (TF-IDF N-gram)|    - Char Obfuscation (leet-speak)      |
|    - Linear SVM (TF-IDF Platt scaled)   |    - Synonym Swap (WordNet)             |
|    x 3 Operating Points:                |    - Ham Word Injection (Good Word)     |
|      * Permissive (FPR <= 0.05)         |    + SentenceTransformer Gate           |
|      * Balanced   (FPR <= 0.01)         |      (Cosine Sim >= 0.75)               |
|      * Strict     (FPR <= 0.001)        |                                         |
|    => 9 Pure Defender Strategies        |    => 6 Attacker Actions                |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. Empirical Payoff Tensor Engine (Stage 4)                                       |
|    Evaluates DR and FPR across all 72 (Type x Defender x Attack) combinations      |
|    Calculates U_D and U_A with c_miss=10, c_FP=100, pi_threat=0.50                 |
|    => U_D, U_A shape: (2 Types x 9 Defenders x 6 Attacks)                          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. Game Solvers & Equilibria (Stage 5)                                            |
|    - Zero-Sum Minimax LP                                                          |
|    - General-Sum Nash (Harsanyi transformation)                                   |
|    - Strong Stackelberg (Brute-Force Enumeration LP)                              |
|    - Bayesian Stackelberg (DOBSS Single-Shot MILP)                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. Validation & Extensions (Stages 6, 8, 9)                                       |
|    - Reality Gap: Model Value vs. 1,000-trial Realized Utility on Unseen Test Data|
|    - Robustness: Threat Prior Sweeps (10%-90%) & Cost Ratios (0.1:1 to 50:1)      |
|    - Budgeted Inspection Game: Tiered Triage (Sum x_d * c_def(d) <= Budget)       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Formal Mathematical Formulation

### 2.1 Players, Types, and Action Spaces
* **Defender Action Space ($D$):** $D = \{0, 1, \dots, 8\}$ representing $9$ pure strategies:
  * $d \in \{0, 1, 2\}$: Naive Bayes at $\tau_{\text{low}}$ (target FPR 0.05), $\tau_{\text{mid}}$ (target FPR 0.01), $\tau_{\text{high}}$ (target FPR 0.001).
  * $d \in \{3, 4, 5\}$: Logistic Regression at $\tau_{\text{low}}, \tau_{\text{mid}}, \tau_{\text{high}}$.
  * $d \in \{6, 7, 8\}$: Linear SVM at $\tau_{\text{low}}, \tau_{\text{mid}}, \tau_{\text{high}}$.
  * Defender mixed strategy: $x \in \Delta(D)$, where $x_d \ge 0$ and $\sum_{d=0}^8 x_d = 1$.

* **Attacker Types ($T$):** Incomplete information modeled via nature selecting type $t \in \{0, 1\}$ according to common prior $p = (p_0, p_1)$:
  * **Type 0 (Bulk Spammer):** Prior $p_0 = 0.70$, Gain $G_0 = 1.0$. Available attacks: $A_0 = \{0, 1, 2, 5\}$ with attack costs $C_0 = \{0.00, 0.05, 0.10, 0.03\}$.
  * **Type 1 (Targeted Phisher):** Prior $p_1 = 0.30$, Gain $G_1 = 5.0$. Available attacks: $A_1 = \{0, 1, 3, 4\}$ with attack costs $C_1 = \{0.00, 0.05, 0.10, 0.15\}$.

* **Attacker Action Space ($A$):** $A = \{0, 1, 2, 3, 4, 5\}$:
  * $a=0$: No Attack
  * $a=1$: Character Obfuscation (strength = 0.15)
  * $a=2$: Synonym Swap (strength = 0.50)
  * $a=3$: Synonym Swap (strength = 1.00)
  * $a=4$: Ham Injection (strength = 1 word)
  * $a=5$: Ham Injection (strength = 3 words)

### 2.2 Payoff Functions

#### Defender Payoff $U_D(t, d, a)$
$$U_D(t, d, a) = -c_{\text{miss}} \cdot (1 - \text{DR}_{d, a}) \cdot \pi_{\text{threat}} - c_{\text{FP}} \cdot \text{FPR}_d \cdot (1 - \pi_{\text{threat}}) - c_{\text{def}}(d)$$
* $c_{\text{miss}} = 10.0$: Penalty for False Negative (unfiltered attack penetrating the gateway).
* $c_{\text{FP}} = 100.0$: Penalty for False Positive (legitimate email incorrectly quarantined).
* $\pi_{\text{threat}} = 0.50$: Base operational threat prevalence.
* $\text{DR}_{d, a} \in [0, 1]$: Empirical Detection Rate of filter $d$ against attack $a$.
* $\text{FPR}_d \in [0, 1]$: Empirical False Positive Rate of filter $d$ on clean ham.
* $c_{\text{def}}(d)$: Computational inspection cost ($0.0$ for NB/LR, $0.1$ for SVM).

#### Attacker Payoff $U_A(t, d, a)$
$$U_A(t, d, a) = G_t \cdot (1 - \text{DR}_{d, a}) - C_t(a)$$
* $G_t$: Gain from successful infiltration ($1.0$ for Spammer, $5.0$ for Phisher).
* $(1 - \text{DR}_{d, a})$: Probability of evasion.
* $C_t(a)$: Attack execution cost. If $a \notin A_t$, $U_A(t, d, a) = -\infty$ (implemented as $-9999$).

---

### 2.3 Game Solvers

#### 1. Bayesian Stackelberg (DOBSS — Decomposed Optimal Bayesian Stackelberg Solver)
Formulated as a single Mixed-Integer Linear Program (MILP):
$$\max_{x, q, v, z} \sum_{t \in T} \sum_{d \in D} \sum_{a \in A_t} p_t \cdot z_{t,d,a} \cdot U_D(t, d, a)$$
subject to:
$$\sum_{d \in D} x_d = 1$$
$$\sum_{a \in A_t} q_{t,a} = 1 \quad \forall t \in T, \quad q_{t,a} \in \{0, 1\}$$
$$v_t - \sum_{d \in D} x_d \cdot U_A(t, d, a) \ge 0 \quad \forall t \in T, a \in A_t$$
$$v_t - \sum_{d \in D} x_d \cdot U_A(t, d, a) \le M(1 - q_{t,a}) \quad \forall t \in T, a \in A_t$$
$$z_{t,d,a} \le x_d, \quad z_{t,d,a} \le q_{t,a}, \quad z_{t,d,a} \ge x_d + q_{t,a} - 1, \quad z_{t,d,a} \ge 0$$
where $z_{t,d,a} = x_d \cdot q_{t,a}$ is the exact McCormick linearization of the bilinear term, and $M$ is a sufficiently large positive constant.

#### 2. Simultaneous Nash Equilibrium (General-Sum)
Solved via the Harsanyi transformation converting the incomplete-information game into an equivalent single normal-form game, using the Lemke-Howson / Support Enumeration algorithm.

#### 3. Zero-Sum Minimax LP
Models a paranoid worst-case baseline where the defender assumes an all-powerful adversary playing zero-sum against defender utility without type priors.

#### 4. Budgeted Inspection Game (Stage 9 Extension)
Incorporates an operational expected computational latency budget:
$$\sum_{d \in D} x_d \cdot c_{\text{def}}(d) \le \text{Budget}$$
enforcing tiered triage across fast/free filters and deep/costly inspection.

---

### 2.4 The Reality Gap Metric
To evaluate whether theoretical equilibria survive deployment against real data:
$$\text{Reality Gap} = |U_D^{\text{model}} - \mathbb{E}[U_D^{\text{realized}}]|$$
where $\mathbb{E}[U_D^{\text{realized}}]$ is computed by running $1,000$ independent Monte Carlo simulation trials on unseen test emails against rational best-responding adversaries, with $95\%$ bootstrap confidence intervals.

---

## 3. Comprehensive Experimental Results

### 3.1 Primary Results: Equilibrium & Reality Gap (Enron vs. Phishing)

#### Enron Dataset (~30,000 emails)
| Game Model | Equilibrium Type | Model Value $U_D$ | Realized Value $U_D$ (95% CI) | Reality Gap | Solve Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Zero-Sum Minimax** | Worst-case, no prior | -0.7241 | -1.0200 [-1.50, -0.60] | 0.2959 | 0.04s |
| **Nash (General-Sum)** | Simultaneous, full prior | -0.6848 | -0.9400 [-1.38, -0.58] | **0.2552** | 50.17s |
| **Strong Stackelberg** | Leader commits, single prior | -0.6848 | -0.7500 [-1.12, -0.49] | **0.0652** | 0.03s |
| **Bayesian Stackelberg (DOBSS)**| Leader commits, hidden types | **-0.6848** | **-0.7500 [-1.12, -0.49]** | **0.0652** | **0.77s** |

#### Phishing Dataset (82,078 emails)
| Game Model | Equilibrium Type | Model Value $U_D$ | Realized Value $U_D$ (95% CI) | Reality Gap | Solve Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Zero-Sum Minimax** | Worst-case, no prior | -1.0358 | -0.6700 [-1.03, -0.37] | 0.3658 | 0.06s |
| **Nash (General-Sum)** | Simultaneous, full prior | -0.9955 | -1.2000 [-1.86, -0.72] | **0.2045** | 36.87s |
| **Strong Stackelberg** | Leader commits, single prior | -0.9955 | -1.0900 [-1.65, -0.63] | **0.0945** | 0.03s |
| **Bayesian Stackelberg (DOBSS)**| Leader commits, hidden types | **-0.9955** | **-1.0900 [-1.65, -0.63]** | **0.0945** | **1.14s** |

*Core Cross-Dataset Finding:*
* **Reality Gap Reduction:** Bayesian Stackelberg cuts the Reality Gap by **74.5% on Enron** (from 0.2552 down to 0.0652) and **53.8% on Phishing** (from 0.2045 down to 0.0945).
* **Computational Speedup:** DOBSS solves in $\approx 1$ second across both datasets, achieving a **32x to 65x speedup** over General-Sum Nash.

---

### 3.2 Classifier Performance & Threshold Calibration (Stage 2)

#### Enron Corpus
| Filter | Target FPR | Calibrated Threshold $\tau$ | Test Detection Rate (DR) | Test False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: |
| Naive Bayes | 0.050 | 0.0037 | 99.79% | 5.00% |
| Naive Bayes | 0.010 | 0.9996 | 87.25% | 1.01% |
| Naive Bayes | 0.001 | inf | 0.00% | 0.00% |
| Logistic Regression | 0.050 | 0.3560 | 99.93% | 4.81% |
| Logistic Regression | 0.010 | 0.6691 | 97.13% | 1.20% |
| Logistic Regression | 0.001 | 0.9299 | 65.54% | 0.06% |
| Linear SVM | 0.050 | 0.0691 | **99.71%** | 3.93% |
| Linear SVM | 0.010 | 0.6026 | **98.64%** | 1.14% |
| Linear SVM | 0.001 | 0.9938 | **73.07%** | 0.06% |

#### Phishing Corpus
| Filter | Target FPR | Calibrated Threshold $\tau$ | Test Detection Rate (DR) | Test False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: |
| Naive Bayes | 0.050 | 0.0015 | 98.30% | 4.31% |
| Naive Bayes | 0.010 | 1.0000 | 71.62% | 0.87% |
| Naive Bayes | 0.001 | inf | 0.00% | 0.00% |
| Logistic Regression | 0.050 | 0.3206 | **99.46%** | 4.51% |
| Logistic Regression | 0.010 | 0.6589 | **96.36%** | 1.07% |
| Logistic Regression | 0.001 | 0.9405 | **73.49%** | 0.05% |
| Linear SVM | 0.050 | 0.1079 | **99.46%** | 4.33% |
| Linear SVM | 0.010 | 0.7307 | **97.18%** | 0.82% |
| Linear SVM | 0.001 | 0.9944 | **67.93%** | 0.05% |

---

### 3.3 Adversarial Evasion Effectiveness (Stage 3)
Evaluated against Logistic Regression at Target FPR = 0.010:

| Adversarial Attack | Attack Strength | Enron Evasion % | Phishing Evasion % | Semantic Validity % |
| :--- | :---: | :---: | :---: | :---: |
| No Attack | 0.0 | 2.0% | 6.0% | 100.0% |
| Character Obfuscation | 0.05 | 2.0% | 5.0% | 100.0% |
| Character Obfuscation | 0.15 | 0.0% | 5.0% | 91.0% - 100.0% |
| Synonym Swap | 0.50 | 2.0% | 7.0% | 100.0% |
| Synonym Swap | 1.00 | 2.0% | 7.0% | 100.0% |
| Ham Injection | 1 word | 13.3% | 7.0% | 98.0% - 100.0% |
| Ham Injection | 3 words | **17.3%** | **12.8%** | 94.0% - 98.0% |

*Key Evasion Takeaway:* Benign ham word injection is the dominant evasion strategy across both datasets, doubling or tripling evasion success compared to synonym swapping or character perturbations.

---

### 3.4 Tiered Triage / Budgeted Inspection Game (Stage 9)

$$\text{Constraint: } \sum_{d \in D} x_d \cdot c_{\text{def}}(d) \le \text{Budget}, \quad c_{\text{def}}(\text{NB})=0.0, c_{\text{def}}(\text{LR})=0.0, c_{\text{def}}(\text{SVM})=0.1$$

| Budget Limit | Enron $U_D$ | Enron SVM Usage | Phishing $U_D$ | Phishing SVM Usage |
| :---: | :---: | :---: | :---: | :---: |
| **0.010** | -0.8857 | **10.0%** | -0.9955 | **0.0%** |
| **0.030** | -0.8754 | **14.4%** | -0.9955 | **0.0%** |
| **0.050** | -0.8246 | **50.0%** | -0.9955 | **0.0%** |
| **0.080** | -0.7407 | **80.0%** | -0.9955 | **0.0%** |
| **0.150** | -0.6848 | **100.0%** | -0.9955 | **0.0%** |

*Core Triage Finding:*
* **Enron:** Linear SVM provides an indispensable accuracy premium on complex corporate text, causing the budget constraint to bind actively. The solver smoothly rations SVM usage from $10\%$ to $100\%$.
* **Phishing:** Logistic Regression achieves $96.36\%$ DR at zero cost ($c_{\text{def}}=0.0$), strictly dominating SVM ($97.18\%$ DR with $-0.10$ cost). DOBSS allocates $100\%$ to Logistic Regression with zero budget consumption, rendering the constraint non-binding.

---

### 3.5 Sensitivity & Robustness Sweeps (Stage 8)

#### Threat Prior Sensitivity (Spammer % vs. Phisher %)
| Spammer % | Phisher % | Enron $U_D$ | Phishing $U_D$ |
| :---: | :---: | :---: | :---: |
| 10% | 90% | -0.4335 | -0.6457 |
| 30% | 70% | -0.5172 | -0.7623 |
| 50% | 50% | -0.6010 | -0.8789 |
| 70% | 30% | -0.6848 | -0.9955 |
| 90% | 10% | -0.7686 | -1.1121 |

#### Cost Ratio Sensitivity ($c_{\text{miss}} : c_{\text{FP}}$)
| Ratio ($c_{\text{miss}} : c_{\text{FP}}$) | Enron $U_D$ | Phishing $U_D$ | Solver Behavioral Adaptation |
| :---: | :---: | :---: | :--- |
| **0.1 : 1** | -0.8084 | -0.9955 | Tolerates misses to avoid costly False Positive penalties |
| **1.0 : 1** | -2.0564 | -1.4888 | Balanced thresholding |
| **10.0 : 1** | -0.9569 | -1.6949 | Transitions toward high-DR strict filtering |
| **50.0 : 1** | -0.9569 | -1.6949 | Permanently locks into maximum-detection filter to avoid miss catastrophe |
