# Raw data

Anonymized Runestone "Datashop" exports for SI 206 / SI 201 (intermediate Python), one folder per semester. **These files are not in git**: they are student activity logs, shared with us by the course for this project. Only `MANIFEST.csv` is committed.

`MANIFEST.csv` lists every export the pipeline reads, the course section it came from, its size and its SHA-256. `so-ell ingest` refuses to run if a file is missing or its checksum changed, so a result can always be traced to the exact bytes it came from.

| Folder | Section files | Notes |
|---|---|---|
| F21, F22, W22, W23, W24 | one combined export | |
| F23, F24, F25 | MW and TTh exported separately | each file numbers students from 1, see design report §4 |
| W20, W21 | one tiny export | 404 and 5 rows; ingested for completeness, excluded from analysis |

To set up on a new machine, copy the semester folders here (compressed `.gz` / `.zip` / `.bz2` files exactly as listed in the manifest) and run `make pipeline`. If you add a semester, drop its export in a new folder, run `uv run so-ell manifest`, add the semester code to `ANALYSIS_SEMESTERS` in `src/so_ell/config.py`, and commit the updated manifest.
