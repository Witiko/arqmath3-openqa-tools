from collections import defaultdict
from statistics import mean
import argparse
import csv
import json
import logging


LOGGER = logging.getLogger(__name__)


def read_task1_result_file(file_path):
    """
    Reading input results file in ARQMath format for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, ranks, and scores
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, answer_id, rank, score, run_name = row
            rank = int(rank)
            score = float(score)
            yield (topic_id, answer_id, rank, score)


def get_top1_task1_results(file_path):
    """
    Reading input results file in ARQMath format for ARQMath Task 3
    and extracting top-1 results
    @param file_path: file path to input file
    @return: iterable of topic ids, and answer ids
    """
    results = defaultdict(lambda: list())
    for topic_id, answer_id, rank, score in read_task1_result_file(file_path):
        results[topic_id].append((score, -rank, answer_id))
    for topic_id, answers in sorted(results.items()):
        *_, answer_id = max(answers)
        yield topic_id, answer_id


def read_task1_qrel_file(file_path):
    """
    Reading input file with relevance judgements for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, and relevance judgements
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, _, document_id, relevance_judgement = row
            relevance_judgement = int(relevance_judgement)
            yield ((topic_id, document_id), relevance_judgement)


def main():
    """
    example: python3 evaluate_task1_results.py -in "TangentS_task1_2021.tsv"
                                               -excluded_topics '[]'
                                               -map "teams_answer_id.tsv"
                                               -qrel "qrel_task1_2022_official.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Compute Task 3 manual evaluation measures (AR, P@1) for Task 1 results')
    parser.add_argument('-in',
                        help='Input result file in ARQMath format for ARQMath Task 1',
                        required=True)
    parser.add_argument('-excluded_topics',
                        help=('A JSON array of topics excluded from the evaluation'),
                        required=True)
    parser.add_argument('-qrel',
                        help='Input file with relevance judgements for ARQMath Task 1',
                        required=True)

    args = vars(parser.parse_args())
    result_file = args['in']
    excluded_topics = args['excluded_topics']
    qrel_file = args['qrel']

    excluded_topics_set = set(json.loads(excluded_topics))
    LOGGER.info(f'Excluded topics: {sorted(excluded_topics_set)}')

    result_dict = dict(get_top1_task1_results(result_file))
    qrel_dict = dict(read_task1_qrel_file(qrel_file))

    missing_topics = set()
    judgements = []
    for topic_id, answer_id in sorted(result_dict.items()):
        if topic_id not in excluded_topics_set:
            continue
        try:
            judgement = qrel_dict[topic_id, answer_id]
            judgements.append(judgement)
        except KeyError:
            missing_topics.add(topic_id)

    if missing_topics:
        LOGGER.warning(f'Results for {len(missing_topics)} topics had no judgements: {sorted(missing_topics)}')
        LOGGER.warning(f'Running the evaluation using just {len(judgements)} topics')

    average_relevance = mean(float(judgement) for judgement in judgements)
    precision_at_one = mean(1.0 if judgement > 1 else 0.0 for judgement in judgements)

    print(f'AR:  {average_relevance:.3f}')
    print(f'P@1: {precision_at_one:.3f}')


if __name__ == "__main__":
    main()
