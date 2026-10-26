# Dataset setup

The analysis expects the workbook at `data/online_retail_II.xlsx`. The local copy is excluded from version control to avoid adding a roughly 45.6 MB binary dataset to the source repository.

Download the Online Retail II dataset from the [official UCI Machine Learning Repository page](https://archive.ics.uci.edu/dataset/502/online+retail+ii), then place the workbook here with the exact filename `online_retail_II.xlsx`.

UCI lists the dataset under the Creative Commons Attribution 4.0 International license. Cite the source when using or redistributing it:

> Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.

The CLI accepts a custom workbook path with `--data-path`. The default is this directory's `online_retail_II.xlsx` file. If the expected workbook is missing, the loader reports its path.
