# Background
This is the second round of data we generated for annotation to judge hierarchical agglomerative clustering models with average linkage.

The community2vec models used to produce these clustering models did not include user profiles pages with subreddits.

## File contents
- `RC_2021-07` & `RC_2022_03`: All raw, unannotated output files associated with the annotation task, including the serialized sklearn agglomerative clustering model used to generate the task and files to be annotated.
- `Agglomerative Coherence Task March 2022_July 2021 All.xlsx`: Results from annotators for the coherence task collected into a single excel file, one sheet per month. Also contains annotation agreement results computed using Excel formulas in a separate sheets.
- `Agglomerative Coherence Task March 2022_July 2021 All - both months raw coherence ratings.csv`: The sheet from `Coherence Task March 2022_July 2021 All.xlsx` that only has coherence labels. Use with `../notebooks/inter_rater_agreements.ipynb` to validate computations from the Excel file.
- `Agglomerative Intruder Task March 2022_July 2021 all.xlsx`: Results from annotators for the intruder task collected into a single excel file, on sheet per month. Each month's sheet includes all annotator's labels and the true answer, with computations for performance.