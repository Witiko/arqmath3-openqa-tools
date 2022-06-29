from collections import defaultdict
import csv
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
