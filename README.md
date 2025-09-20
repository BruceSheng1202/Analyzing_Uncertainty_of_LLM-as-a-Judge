
# README

## 📌 Project Overview

This repository contains the experiments and analysis for our research on **Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction**.


**Abstract**: LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment.




---

## 📂 Repository Structure and Description

Example_GenAI-Bench: the experiments on GenAI-Bench as an example to use our framework
```
1. VQA_eval.py: evaluation code covering evaluation by a VLM, prompt design with image and text, obatin response, token targeting, logits extraction and record saving

2. evaluation_metrics.py: to calcualte the metrics that used to evaluate the midpoints (correlations and errors)

3. interval_processing.py: range clipping, boundary adjustment and calculating coverage and width

4. performance.ipynb: present the results of application on GenAI-Bench, where you can see a csv file keeping responses, logits, scores and intervals etc.
```

Analysis: performance evaluation to form the tables and analysis in the paper
```
1. R2CCP_distribution_shift.ipynb: summeval and dialsumm are used to calibrate each other

2. R2CCP_validity.ipynb: an example that two same evaluation task but get different scores due to temperature

3. calsize_comparison.ipynb and calsize_experiment.ipynb: to generate and compare intervals with different calibration size

4. heteroskedaticity_ht.ipynb: hypothesis testing for  heteroskedaticity

5. human_performance.ipynb: the construction of human_based baseline

6. oversampling_raws.ipynb: to evaluate the performance of oversampling and majority vote

7. prompt_sensitivity.ipynb: to compare the performance between GPT-4o and GPT-4o mini, and different prompts

8. score_performacne.py: to calculate the performance of scores and midpoints

9. statistics_intervals.ipynb: to calculate the performance of intervals

10. temperature_comparison.ipynb: to compare the interval performance with different judge models and temperature 
```

conformal_predictors: 9 methods keeped in 7 files, where CQR and BoostedCP each has 2 variants


evaluations and reprompt on server: to evaluate by qwen and dsr1 on server, reprompt and regrade with analysis
```
1. qwen_eval.py: to evaluate on summeval

2. qwen_eval_dialsumm.py: to evaluate on dialsumm

3. reasoning_eval.py: to evaluate on roscoe

4. reasoning_eval_oversampling: to evaluate on roscoe by oversampling

5. reprompt_analysis.py: to analyze why judge model resist to change score

6. reprompt_improvement.py: to evaluate the improvement of reprompting

7. reprompt_regrade.py: to reprompt and regrade on summeval

8. reprompt_regrade_reasoning.py: to reprompt and regrade on roscoe

9. 3 bash code to run evaluations and 3 prompts for reprompt_analysis.py
```

---
