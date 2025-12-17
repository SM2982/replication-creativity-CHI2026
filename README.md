# Creativity Experiment: AI Chatbot Interaction Analysis

Analysis code and data for two experimental studies investigating how different types of AI chatbot interactions affect creative idea generation, idea diversity, and perceived ownership.

- **Study 1:** Main study (Alternative Uses Test)
- **Study 2:** Validation study in different context

## Repository Structure

- `analysis/Study1/` - Study 1 analysis (main)
  - `analysisR/Analysis_Study.Rmd` - R Markdown analysis file
- `analysis/Study2/` - Study 2 analysis (validation)
  - `Code/Analysis_Study_new.Rmd` - R Markdown analysis file
  - `Data/` - Anonymized datasets
  - `Figures/` - Generated visualizations
- `analysis/rater/` - Rater documentation

---

# Study 1 (Main Study)

## Experimental Conditions
- **Question-Mode** - AI asks clarifying questions
- **Suggestion-Mode** - AI provides direct suggestions  
- **Model-Led** - AI leads the creative process
- **Vanilla** - Basic AI interaction
- **Control** - No AI assistance

Only in the validation study:  **Model-Led (iterative)**

## Key Measures
- Idea Quality (expert ratings)
- Idea Diversity (semantic embeddings)
- Perceived Ownership


## Running the Analysis

**Prerequisites:** R 4.0+, RStudio

1. Open `analysis/Study1/analysisR/Analysis_Study.Rmd` in RStudio
2. Run all chunks (required R packages install automatically)

The analysis performs:
- Mixed-effects models with planned contrasts
- Reliability analyses (Cronbach's α, ICC)
- Robustness checks (ordinal models, non-parametric tests)
- Publication-ready visualizations

---

# Study 2 (Validation Study)

Validation study in a different context.

## Running the Analysis

1. Open `analysis/Study2/Code/Analysis_Study_new.Rmd` in RStudio
2. Run all chunks

**Outputs:** Figures saved to `analysis/Study2/Figures/`

---


## Data Privacy & Ethics

- All data is anonymized
- Consent revocations and data expiration handled automatically
- IRB-approved studies

## Citation

```
[Publication details forthcoming]
```

## Reproducibility Statement

All analyses in this repository are fully reproducible. The R Markdown files contain complete analysis code with automatic package installation. Running the analysis scripts will regenerate all statistical results, figures, and tables from the anonymized datasets.
