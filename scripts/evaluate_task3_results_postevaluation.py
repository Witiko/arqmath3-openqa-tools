from pathlib import Path
from statistics import mean
import json
import argparse
import csv
import logging


LOGGER = logging.getLogger(__name__)


def read_task3_result_file(file_path):
    """
    Reading input results file in ARQMath format for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, *_ = row
            yield topic_id


def read_task3_map_file(file_path, expected_run_name):
    """
    Reading map file from topic IDs and run names to document IDs for ARQMath Task 3
    @param file_path: file path to input file
    @param expected_run_name: run name of the currently evaluated results
    @return: iterable of topic ids and answer ids
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            run_name, topic_id, answer_id = row
            if run_name == expected_run_name:
                yield (topic_id, answer_id)


def read_task3_postevaluation_qrel_file(file_path):
    """
    Reading input file with post-evaluation assessments for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, whether the answers are machine generated,
             and whether the answers contain unrelated information
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        next(csv_reader)
        for row in csv_reader:
            topic_id, answer_id, is_machine_generated, has_unrelated_information = row
            is_machine_generated = int(is_machine_generated) == 1
            has_unrelated_information = int(has_unrelated_information) == 1
            yield (topic_id, answer_id), (is_machine_generated, has_unrelated_information)


def main():
    """
    example: python3 evaluate_task3_results_postevaluation.py
               -in "Baseline2022-task3-GPT3-auto-both-generate-P.tsv"
               -excluded_topics '["A.367"]'
               -map "teams_answer_id.tsv"
               -qrel "task3-extra-assessment.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Compute Task 3 post-evaluation measures (MG, UI) for Task 3 results')
    parser.add_argument('-in',
                        help='Input result file in ARQMath format for ARQMath Task 3',
                        required=True)
    parser.add_argument('-excluded_topics',
                        help='A JSON array with topics excluded from the evaluation',
                        required=True)
    parser.add_argument('-map',
                        help='Input map file from topic IDs and run names to synthetic answer IDs for ARQMath Task 3',
                        required=True)
    parser.add_argument('-qrel',
                        help=('Input file with post-evaluation assessments for ARQMath Task 3'),
                        required=True)

    args = vars(parser.parse_args())
    result_file = args['in']
    excluded_topics = args['excluded_topics']
    run_name = Path(result_file).stem
    map_file = args['map']
    qrel_file = args['qrel']

    excluded_topics_set = set(json.loads(excluded_topics))
    LOGGER.info(f'Excluded topics: {sorted(excluded_topics_set)}')

    map_dict = dict(read_task3_map_file(map_file, run_name))
    qrel_dict = dict(read_task3_postevaluation_qrel_file(qrel_file))

    missing_topics = set()
    result_dict = dict()
    for topic_id in read_task3_result_file(result_file):
        if topic_id in excluded_topics_set:
            continue
        if topic_id in result_dict:
            raise ValueError(f'Repeated topic {topic_id} in {result_file}')
        try:
            answer_id = map_dict[topic_id]
            result_dict[topic_id] = answer_id
        except KeyError:
            missing_topics.add(topic_id)

    missing_topics = set()
    are_machine_generated = []
    have_unrelated_information = []
    for topic_id, answer_id in sorted(result_dict.items()):
        try:
            is_machine_generated, has_unrelated_information = qrel_dict[topic_id, answer_id]
            are_machine_generated.append(is_machine_generated)
            have_unrelated_information.append(has_unrelated_information)
        except KeyError:
            missing_topics.add(topic_id)

    if missing_topics:
        LOGGER.warning(f'Results for {len(missing_topics)} topics had no judgements: {sorted(missing_topics)}')
        LOGGER.warning(f'Running the evaluation using just {len(are_machine_generated)} topics')

    machine_generated_ratio = mean(float(judgement) for judgement in are_machine_generated)
    unrelated_information_ratio = mean(float(judgement) for judgement in have_unrelated_information)

    print(f'MG: {machine_generated_ratio:.3f}')
    print(f'UI: {unrelated_information_ratio:.3f}')


if __name__ == "__main__":
    main()
