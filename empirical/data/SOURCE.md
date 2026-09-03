# Data provenance

Dataset: Buchwald–Hartwig high-throughput reaction yields associated with:

> Ahneman, D. T., Estrada, J. G., Lin, S., Dreher, S. D. & Doyle, A. G.
> Predicting reaction performance in C–N cross-coupling using machine learning.
> *Science* 360, 186–190 (2018). DOI: `10.1126/science.aar5169`.

Downloaded workbook:

`https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`

- Workbook SHA-256: `d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e`
- Selected worksheet: `Plates1-3`
- Derived CSV SHA-256: `a7c415d9ff58d940cc8214c46fa4af12fd7a588f0b4314d0798e70f7a7eb2083`
- Rows: 3,955 measured reactions plus one header row
- Columns: `Ligand, Additive, Base, Aryl halide, Output`

The CSV is a value-preserving worksheet export. No row was imputed, normalized,
or deduplicated. The repository does not assert a new license over the upstream
data; users should verify the upstream terms for redistribution and publication.

