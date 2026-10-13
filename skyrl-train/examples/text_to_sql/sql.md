## Downloading SQL Data

We provide the training set we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data-newfmt 

You can download the dataset by running the following command:

```bash
huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
```

## DB environment 

Next, download the data used to populate the database. We use the database files from [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/README.md):


- [ModelScope-OmniSQL-datasets](https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary)
- [HuggingFace-OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets)


The dataset download is 22.2GB and include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

To download, run:

```bash
huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir $HOME/data/sql/db_files/

unzip $HOME/data/sql/db_files/data.zip -d $HOME/data/sql/db_files/
```

## Prepare TaskSpecs

The downloads contain raw source rows, not training inputs. Convert each split before training:

```bash
uv run --project .. python -m marinskyrl.task_preparation \
  --input raw/train.parquet --output prepared/train.parquet --source skyrl_sql \
  --config sql-grader.json --environment python-environment.json
```

Use the same command for the validation split. `sql-grader.json` contains
`db_path`: the database root on each rollout worker. The session reads the private
reference database there and sends a database copy to the task machine.
Make that path accessible on every rollout worker before launch.

`python-environment.json` declares `command_semantics="linux_process"` and a
prebuilt Python `docker_image` with an `@sha256` digest. The task keeps database
identifiers, reference SQL, and grader parameters in private verifier data.
The run config selects the `text2sql` session and execution limits.
The training scripts request 1 CPU and 8192 MB for each task machine.

## Training

Set `DATA_DIR` in the training script to the prepared TaskSpec directory.

To start training, run:

```bash
bash examples/text_to_sql/run_skyrl_sql.sh
```
