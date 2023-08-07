import os
import json
import argparse
import datasets
from tqdm import tqdm
import concurrent.futures
from promptsource.templates import DatasetTemplates


def export_dataset(
    dataset_output_dir,
    dataset_name,
    subset_name,
    prompt_template,
    prompt,
    dataset,
):
    splits = list(dataset.keys())
    prompt_name = prompt.get_name()
    for split in splits:
        dataset_split = dataset[split]
        json_data_path = os.path.join(dataset_output_dir, split)
        os.makedirs(json_data_path, exist_ok=True)
        json_data_path = os.path.join(
            json_data_path,
            (prompt_template + "." + prompt_name).replace("/", "_").replace(" ", "_")
            + ".jsonl",
        )
        with open(json_data_path, "w", encoding="utf-8") as file_ptr:
            total_num_sample = len(dataset_split)
            for _id, sample in tqdm(
                enumerate(dataset_split),
                total=total_num_sample,
                desc="{}_{}_{}_{}_{}".format(
                    dataset_name, subset_name, split, prompt_template, prompt_name
                ),
            ):
                projected_sample = prompt.apply(sample)
                if len(projected_sample) != 2:
                    continue
                source, target = projected_sample
                projected_sample_with_metadata = {
                    "id": _id,
                    "source": source,
                    "target": target,
                    "prompt_template": prompt_template,
                    "prompt_name": prompt_name,
                    "dataset_name": dataset_name,
                    "subset_name": subset_name,
                    "split": split,
                }
                file_ptr.write(json.dumps(projected_sample_with_metadata))
                file_ptr.write("\n")
    return "Completed:: {}!".format(json_data_path)


def project_prompt_dataset(
    raw_output_dir, dataset_name, subset_name, prompt_template, cache_dir, square_root_num_proc=1
):
    dataset = datasets.load_dataset(dataset_name, subset_name, cache_dir=cache_dir)
    if prompt_template is None:
        if subset_name is None:
            prompt_template = "{}".format(dataset_name)
        else:
            prompt_template = "{}/{}".format(dataset_name, subset_name)
    dataset_output_dir = os.path.join(raw_output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    if subset_name is not None:
        dataset_output_dir = os.path.join(dataset_output_dir, subset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

    prompts = DatasetTemplates(prompt_template)
    prompt_names = list(prompts.name_to_id_mapping.keys())
    # for prompt_name in prompt_names:
    #     prompt = prompts[prompt_name]
    #     export_dataset(
    #         dataset_output_dir,
    #         dataset_name,
    #         subset_name,
    #         prompt_template,
    #         prompt,
    #         dataset,
    #     )
    total_num_prompts = len(prompt_names)
    with concurrent.futures.ProcessPoolExecutor(max_workers=square_root_num_proc) as executor:
        for _out in tqdm(
            executor.map(
                export_dataset,
                [dataset_output_dir for _ in range(total_num_prompts)],
                [dataset_name for _ in range(total_num_prompts)],
                [subset_name for _ in range(total_num_prompts)],
                [prompt_template for _ in range(total_num_prompts)],
                [prompts[prompt_name] for prompt_name in prompt_names],
                [dataset for _ in range(total_num_prompts)],
            ),
            total=total_num_prompts,
        ):
            try:
                print(_out)
            except Exception as emsg:
                print("Exception msg: {}".format(emsg))


def invoke_none(lst):
    for idx, val in enumerate(lst):
        if val == "None" or val == "none" or val == "null" or val == "":
            lst[idx] = None
    return lst


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name-or-paths",
        nargs="+",
        default="glue",
        help="Path to the dataloader file",
    )
    parser.add_argument(
        "--dataset-configs",
        nargs="+",
        default=None,
        help="Config of the dataset split.",
    )
    parser.add_argument(
        "--prompt-templates-configs",
        nargs="+",
        default=None,
        help="Config of the prompt templates.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to the cache dir. (The directory may require very large space if it's not cached earlier.)",
    )
    parser.add_argument(
        "--raw-output-dir", type=str, required=True, help="Path to the raw-output dir."
    )
    parser.add_argument(
        "--square-root-num-proc",
        type=int,
        default=9,
        help="Total number of parallel process will be `--square-root-num-proc * --square-root-num-proc.`",
    )
    args = parser.parse_args()

    assert len(args.dataset_name_or_paths) == len(args.dataset_configs)
    assert len(args.dataset_name_or_paths) == len(args.prompt_templates_configs)

    invoke_none(args.dataset_name_or_paths)
    invoke_none(args.dataset_configs)
    invoke_none(args.prompt_templates_configs)

    # for (dataset_name_or_path, dataset_config, prompt_template_config) in zip(
    #     args.dataset_name_or_paths, args.dataset_configs, args.prompt_templates_configs
    # ):
    #     project_prompt_dataset(
    #         args.raw_output_dir,
    #         dataset_name_or_path,
    #         dataset_config,
    #         prompt_template_config,
    #         cache_dir=args.cache_dir,
    #         square_root_num_proc=args.square_root_num_proc,
    #     )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.square_root_num_proc) as executor:
        for _out in tqdm(
            executor.map(
                project_prompt_dataset,
                [args.raw_output_dir for _ in range(len(args.dataset_name_or_paths))],
                [dataset_name_or_path for dataset_name_or_path in args.dataset_name_or_paths],
                [dataset_config for dataset_config in args.dataset_configs],
                [prompt_template_config for prompt_template_config in args.prompt_templates_configs],
                [args.cache_dir for _ in range(len(args.dataset_name_or_paths))],
                [args.square_root_num_proc for _ in range(len(args.dataset_name_or_paths))],
            ),
            total=len(args.dataset_name_or_paths),
        ):
            try:
                print(_out)
            except Exception as emsg:
                print("Exception msg: {}".format(emsg))

if __name__ == "__main__":
    main()
