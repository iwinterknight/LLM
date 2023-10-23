import os
import argparse
import yaml

from data_preprocessing import process_data
from train import train

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--projectdir', dest='projectdir', type=str, help='Name of the project directory')
    # args = parser.parse_args()
    # projectdir = args.projectdir
    projectdir = "C:/Users/thewi/Documents/Projects/Taskbot_Challenge/TacoQA_Alt"

    with open(os.path.join(projectdir, "config.yaml"), "r") as stream:
        config = yaml.safe_load(stream)

    tokenized_dataset, tokenizer = process_data(projectdir, config)
    train(tokenized_dataset, projectdir, config)


if __name__ == '__main__':
    main()
