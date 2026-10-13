# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
import shutil

from marinskyrl.task_format import TASK_SPEC_COLUMN
from marinskyrl.task_preparation import source_row_task
from skyrl_gym.tools.search import DEFAULT_TIMEOUT

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

_HDFS_PREFIX = "hdfs://"

_HDFS_BIN_PATH = shutil.which("hdfs")


def exists(path: str, **kwargs) -> bool:
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    """hdfs capable to check whether a file_path is exists"""
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def copy(src: str, dst: str, **kwargs) -> bool:
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def _run_cmd(cmd: str, timeout=None):
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_non_local(path: str):
    return path.startswith(_HDFS_PREFIX)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and "
    "</information>. You can search as many times as you want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def process_single_row(row, current_split_name, row_index, *, config):
    """Prepare a search TaskSpec with private answers and tool parameters."""
    question = row.get("question", "")

    # Build prompt structure
    user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + question
    prompt = [{"role": "system", "content": DEFAULT_SYSTEM_CONTENT}, {"role": "user", "content": user_content}]

    # Extract ground truth from the "reward" field or fallback to golden_answers
    reward_data = row.get("reward_model")
    if isinstance(reward_data, dict) and "ground_truth" in reward_data:
        ground_truth = reward_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    task = source_row_task(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_spec": reward_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        },
        row_index,
        source_name=data_source_tagged,
        config=config,
    )
    return {TASK_SPEC_COLUMN: task.model_dump_json()}


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    # Determine splits to process
    splits = [args.split] if args.split else ["train", "test"]

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in splits:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                # Download Parquet file from HuggingFace
                logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                local_parquet_filepath = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=parquet_filename,
                    repo_type="dataset",
                    local_dir=tmp_download_dir,
                    local_dir_use_symlinks=False,
                )

                # Load and process Parquet file
                raw_table = pq.read_table(local_parquet_filepath)
                if args.max_rows is not None:
                    limit = args.max_rows if split == "train" else min(args.max_rows, 10)
                    raw_table = raw_table.slice(0, limit)

                logger.info(f"Loaded {raw_table.num_rows} rows from {parquet_filename}")
                prepared = [
                    process_single_row(
                        row,
                        current_split_name=split,
                        row_index=index,
                        config={
                            "search_url": args.search_url,
                            "topk": args.topk,
                            "timeout": args.timeout,
                            "log_requests": args.log_requests,
                        },
                    )
                    for index, row in enumerate(raw_table.to_pylist())
                ]

                if split == "test":
                    logger.info("Saving `test.parquet` from HF as `validation.parquet` locally")
                    output_file_path = os.path.join(local_save_dir, "validation.parquet")
                else:
                    output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                pq.write_table(pa.Table.from_pylist(prepared), output_file_path)
                logger.info(f"Saved {len(prepared)} prepared tasks to {output_file_path}")
                processed_files.append(output_file_path)

            except EntryNotFoundError:
                logger.warning(f"{parquet_filename} not found in repository {args.hf_repo_id}")

    if not processed_files:
        raise ValueError("No source data matched the selected splits")

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        logger.info(f"Copied prepared tasks to HDFS: {args.hdfs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir", default="./data/searchR1", help="Local directory to save the processed Parquet files."
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process from each split.")
    parser.add_argument("--search_url", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--log_requests", action="store_true")
    parser.add_argument(
        "--split", choices=["train", "test"], default=None, help="If set, only download this split (train or test)."
    )

    args = parser.parse_args()

    main()
