# Self-reported Demographics and Discourse Dynamics in a Persuasive Online Forum

This repository contains all the scripts and data used in the paper "Self-reported Demographics and Discourse Dynamics in a Persuasive Online Forum", (Falenska, Vecchi, & Lapesa, LREC-COLING 2024).

## Data

- `data/lrec2024_data.csv` full data frame with all our features and annotations
- `manual_annotations` explicit gender annotations provided by human annotators

## Code

- `scripts/analyze/analyze_stats.ipynb` frequency and topic analysis
- `scripts/analyze/regression_analysis.Rmd` all the regression analysis
- `scripts/explicit/highlight-genders.sh` heuristics for filtering and highlighting explicit gender mentioned. 

## References

When using any of the data provided in this repository, please cite the following papers:

* Agnieszka Falenska, Eva Maria Vecchi, and Gabriella Lapesa. 2024. **Self-reported Demographics and Discourse Dynamics in a Persuasive Online Forum**. In *Proceedings of the International Language Resources and Evaluation (LREC-COLING-2024)*. Torino, Italy. 28-30 May 2024.
* Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil, and Lillian Lee. 2016. **Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions**. In Proceedings of the 25th International Conference on World Wide Web, WWW ’16, page 613–
624, Republic and Canton of Geneva, CHE.
