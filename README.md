# Hate Speech Detection

## Data Preparation

All the scripts in the following section for preprocessing can be run by just runniung `hate_speech_pipeline/preprocessing/main.py`. This is small runner that executes the preprocessing scripts in order. It will call the scripts found in the scripts directory and forward the selected DB, lexicon CSV and output directory to the relevant steps.

| Argument        | Type | Default        | Description                                                                                 |
| --------------- | ---- | -------------- | ------------------------------------------------------------------------------------------- |
| `--scripts-dir` | path | this folder    | Directory containing the preprocessing scripts (defaults to the same folder as the runner). |
| `--db`          | path | required       | Path to the sqlite DB to operate on (required).                                             |
| `--lexicon-csv` | path | -              | Path to `refined_ngram_dict.csv` for `make_train_pool_from_lexicon.py` (optional).          |
| `--outdir`      | path | `prepared/csv` | Output directory used by `dump_supervision_threads.py`.                                     |

Note, this will use all default args for the scripts it runs, refer to the individual script details in the following section.

#### Individual Pre-processing Scripts

#### Ingest .zst archives into SQLite - `preprocessing/build_db_from_zst.py`

Stream large Reddit .zst dumps (submissions and comments) into a local SQLite database and create base tables and indexes. Writes to the DB (default `reddit_2016_11.db`).

#### Remove orphan comments - `preprocessing/fast_prune_orphans_cte.py`

Remove orphan comments that are not connected to any submission (prune unreachable threads) using a recursive CTE, then VACUUM the DB.

- Usage: `python fast_prune_orphans_cte.py <db>`
- Arguments:

  | Argument | Type | Default             | Descriptio                     |
  | -------- | ---- | ------------------- | ------------------------------ |
  | db       | path | `reddit_2016_11.db` | Path to the sqlite DB to prune |

#### Build a lexicon-matched pool - `preprocessing/make_train_pool_from_lexicon.py`

Scan train comments for occurrences of the Davidson lexicon n-grams and populate a `train_lex_pool` table of matching comment ids.

- Arguments:

  | Argument  | Type | Default  | Description                                  |
  | --------- | ---- | -------- | -------------------------------------------- |
  | `--db`    | path | required | Path to the sqlite DB.                       |
  | `--csv`   | path | required | Lexicon CSV (e.g. `refined_ngram_dict.csv`). |
  | `--batch` | int  | 100000   | Number of rows to fetch per DB chunk.        |

#### Build supervision training tables with matched negatives - `preprocessing/build_supervision_train.py`

Create `positives_train` and a balanced `supervision_train` table with matched negatives (thread & time matched).

- Arguments:

  | Argument           | Type           | Default        | Description                                                                     |
  | ------------------ | -------------- | -------------- | ------------------------------------------------------------------------------- |
  | `--db`             | path           | required       | Path to the sqlite DB.                                                          |
  | `--train-end`      | unix timestamp | script default | Default: training ends at 2016-11-20 UTC                                        |
  | `--val-end`        | unix timestamp | script default | End of validation range                                                         |
  | `--window-seconds` | int            | 86400          | Time window for negative sampling (seconds).                                    |
  | `--k`              | int            | 1              | Negatives per positive.                                                         |
  | `--make-val-test`  | flag           | False          | Create `supervision_val` and `supervision_test` tables with natural prevalence. |

#### Split supervision into train/val/test by thread chronologically - `preprocessing/chrono_thread_split_80_10_10.py`

Produce a thread-aware chronological split of a supervision table into train/val/test by thread start time, keeping entire threads together.

- Arguments:

  | Argument           | Type   | Default               | Descriptio                              |
  | ------------------ | ------ | --------------------- | --------------------------------------- |
  | `--db`             | path   | required              | Path to the sqlite DB.                  |
  | `--train-table`    | string | `supervision_train`   | Source supervision table.               |
  | `--comments-table` | string | `comments`            | Comments table name.                    |
  | `--train-out`      | string | `supervision_train80` | Output table name for train split.      |
  | `--val-out`        | string | `supervision_val10`   | Output table name for validation split. |
  | `--test-out`       | string | `supervision_test10`  | Output table name for test split.       |
  | `--train-ratio`    | float  | 0.8                   | Train split ratio.                      |
  | `--val-ratio`      | float  | 0.1                   | Validation split ratio.                 |

### Export thread CSVs for modeling - `preprocessing/dump_supervision_threads.py`

Export full thread-level CSVs for each supervision split (train/val/test). Each CSV contains all comments in threads that contain supervision ids and includes the label for supervised ids.

- Arguments:

  | Argument           | Type   | Default               | Descriptio                     |
  | ------------------ | ------ | --------------------- | ------------------------------ |
  | `--db`             | path   | required              | Path to the sqlite DB.         |
  | `--outdir`         | path   | `prepared/csv`        | Directory to write CSVs.       |
  | `--comments-table` | string | `comments`            | Comments table to export from. |
  | `--train-split`    | string | `supervision_train80` | Table name for train split.    |
  | `--val-split`      | string | `supervision_val10`   | Table name for val split.      |
  | `--test-split`     | string | `supervision_test10`  | Table name for test split.     |

## Modeling

### Command Line Arguments

| Argument              | Type   | Default                    | Description                                           |
| --------------------- | ------ | -------------------------- | ----------------------------------------------------- |
| --mode                | str    | "diffusion"                | One of `"diffusion"` or `"classification"`.           |
| --generate-embeddings | flag   | False                      | Don't assume embeddings are available, generate them. |
| --train-file-path     | string | train_dataset_with_emb.csv | Path to the training CSV file.                        |
| --test-file-path      | string | test_dataset_with_emb.csv  | Path to the test CSV file.                            |
| --val-file-path       | string | val_dataset_with_emb.csv   | Path to the validation CSV file.                      |
| --subset-count        | int    | 500                        | Number of samples to use from the dataset.            |
| --window-size-hours   | int    | 1                          | Number of hours to use for snapshot window.           |
| --epochs              | int    | 10                         | Number of epochs to train for.                        |

### Sample Commands

#### Diffusion

1. Look at a subset of records, **recommended to run this first**.

   ```
   python main.py \
   --train-file-path retrain_train80.csv \
   --test-file-path retrain_test10.csv \
   --val-file-path retrain_validation10.csv \
   --subset-count=1000 \
   --generate-embeddings
   ```

1. If no embeddings available, generate them.

   ```
   python main.py \
   --train-file-path retrain_train80.csv \
   --test-file-path retrain_test10.csv \
   --val-file-path retrain_validation10.csv \
   --generate-embeddings
   ```

1. Files assumed to contain embeddings, no need to generate them. This is for subsequent runs, when files with embeddings are already saved.

   ```
   python main.py \
   --train-file-path retrain_train80_with_embeddings.csv \
   --test-file-path retrain_test10_with_embeddings.csv \
   --val-file-path retrain_validation10_with_embeddings.csv \
   ```

#### Node Classification

```
python main.py \
 --mode classification \
 --train-file-path retrain_train80_with_embeddings.csv \
 --test-file-path retrain_test10_with_embeddings.csv \
 --val-file-path retrain_validation10_with_embeddings.csv \
 --epochs 20
```
