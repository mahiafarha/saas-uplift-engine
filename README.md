# SaaS Causal Uplift & Prescriptive Retention Engine

An end-to-end causal machine learning pipeline and production microservice designed for SaaS subscription retention. Rather than predicting raw churn probability, this system models **Individual Treatment Effects (ITE)** using meta-learners to identify *Persuadable* accounts and prevent outreach to *Sleeping Dogs*.

---

## 📌 Executive Summary & Financial Impact

Traditional predictive churn models suffer from a fatal business flaw: **high churn risk does not mean a user is savable**. Sending discounts to all high-risk accounts wastes budget on users who leave regardless ("Lost Causes") or actively accelerates churn among dormant accounts ("Sleeping Dogs").

* **Naive Policy (Target All at Risk):** Projected net benefit of **+$37,812.69**.
* **Prescriptive Causal Policy (Target by Expected ROI):** Generates **+$97,162.47** in net value.
* **Efficiency Advantage:** **+$59,349.78 (+156% increase in ROI)** on the exact same test cohort of 2,000 accounts.

---

## 🏗️️ Architecture & Technical Stack

* **Data Engineering & Simulation:** Synthetic event logs simulating counterfactual experimental outcomes across tiers (`pandas`, `numpy`).
* **Causal Modeling:** Two-Model Meta-Learner (T-Learner) leveraging `HistGradientBoostingClassifier` trained on counterfactual branches ($M_0$ and $M_1$).
* **Evaluation Framework:** Cumulative Qini Curves, uplift quintiles, and prescriptive financial threshold optimization.
* **Production Serving:** Asynchronous REST microservice built with `FastAPI` and `Pydantic` issuing automated actions (`TARGET_WITH_OFFER`, `DO_NOT_DISTURB`, `PASSIVE_MONITORING`).
* **Testing & Deployment:** Integration testing via `pytest` and containerization with `Docker` & `docker-compose`.

---

## 📊 Methodology: Uplift Quadrants

$$\text{Uplift } \tau(X) = P(\text{Churn} \mid T=0, X) - P(\text{Churn} \mid T=1, X)$$

| Quadrant | Baseline Risk | Treatment Effect | Action Prescribed |
| :--- | :--- | :--- | :--- |
| **Persuadables** | High | Significant Churn Reduction | `TARGET_WITH_OFFER` |
| **Sleeping Dogs** | Moderate/High | Negative Lift (Outreach accelerates churn) | `DO_NOT_DISTURB` |
| **Lost Causes** | Very High | Negligible / Zero Lift | `PASSIVE_MONITORING` |
| **Sure Things** | Low | Negligible / Zero Lift | `PASSIVE_MONITORING` |

---

## 🚀 Quickstart & Reproduction

### Option A: Run via Docker Compose
```bash
docker compose up --build