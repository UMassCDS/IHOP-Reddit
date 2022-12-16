# Background
This is the first round of data we generated for annotation. It was generated in the same manner as the KMeans `prep_annotation_data` targets in the DVC config.

Because of this task, we noticed the user profile pages appearing with the subreddits. We fixed that issue in data pre-processing for later models. However, the results for this annotation task are still valid, as removing a few user profile pages from the clusterings will not hurt coherence or interpretability. These annotations are kept for transparency, completeness and to avoid repeating the task. Any future clusterings will be at least as good as this one.

## File contents
- `RC_2021-07` & `RC_2022_03`: All raw, unannotated output files associated with the annotation task, including the serialized KMeans++ model used to generate the task and files to be annotated.
- `Coherence Task March 2022_July 2021 All.xlsx`: Results from annotators for the coherence task collected into a single excel file, one sheet per month. Also contains annotation agreement results computed using Excel formulas in a separate sheets.
- `Coherence Task March 2022_July 2021 All - both months raw labels.csv`: The sheet from `Coherence Task March 2022_July 2021 All.xlsx` that only has coherence labels. Use with `../notebooks/inter_rater_agreements.ipynb` to validate computations from the Excel file.
- `Intruder Task March 2022_July 2021 all.xlsx`: Results from annotators for the intruder task collected into a single excel file, on sheet per month. Each month's sheet includes all annotator's labels and the true answer, with computations for performance. `Intruder Task March 2022_July 2021 all - 2021-07 - all.csv` and `Intruder Task March 2022_July 2021 all - 2022-03 - all.csv` are just the sheets from this Excel file exported as CSV for convenience.
