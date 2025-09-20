
# README

## Project Overview

This repository contains the experiments and analysis for our research on **Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction**.


**Abstract**: LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment.




---

## 📂 Repository Structure

```
Example_GenAI-Bench/     # Example benchmark tasks for evaluation
LVD/                     # LVD predictor
boosted-conformal/       # Boosted CP predictor
calsize/                 # Calibration size intervals
chr/                     # CHR predictor
distribution_shift/      # Tests under distribution shift: summeval vs dialsumm
evaluations/             # Evaluation materials for DSR1, QWEN and reprompting  
evaluations/             # Evaluation materials for GPT-4o and GPT-4o mini 
human_performance/       # Human-based baseline construction and comparisons
interval_results/        # Interval evaluation results
model_logits/            # Extracted Logit
oversampling/            # Oversampling experiments, comare with repromppt
prompt_sensitivity/      # Prompt variation experiments 
raw_scores/              # Raw scores from evaluations
run_cost/                # Cost efficiency analysis
temperature/             # Effect of temperature on uncertainty
validity/                # A special example that two same evaluations with different score 
```

Jupyter notebooks for running experiments:

* Different conformal predictors: `BoostedCP_random.ipynb`, `CHR_random.ipynb`, `CQR_random.ipynb`, `R2CCP_random.ipynb`, `CQR_random.ipynb`, `OrdinalAPS_random.ipynb`, `OrdinalRC_random.ipynb`
* Analysis: `calsize_experiment.ipynb`, `plot_calsize_instances.ipynb`, `score_performance.ipynb`, `heteroskedasticity_ht.ipynb`

Figures:

* `instance_before_adjustment.png`, `instance_after_adjustment.png`

---
