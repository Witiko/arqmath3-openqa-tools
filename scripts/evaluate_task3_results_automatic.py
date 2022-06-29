import argparse
from collections import defaultdict
import csv
import logging
from pathlib import Path

from arqmathcode.post_reader_record import DataReaderRecord
from lxml import etree
from lxml.html import document_fromstring
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)


def read_task1_result_file(file_path):
    """
    Reading input results file in ARQMath format for ARQMath-3 Task 3
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


def convert_task1_answer_id_to_answer(answer_id, data_reader_record):
    """
    Converting answer ids to answer bodies in text + LaTeX format
    @param answer_id: id of the answer
    @param data_reader_record: ARQMathCode data reader
    @return the body text of the answer in text + LaTeX format
    """
    answer = data_reader_record.post_parser.map_just_answers[answer_id]
    answer_body = answer.body
    try:
        parsed_answer_body = document_fromstring(answer_body)
    except etree.ParserError:
        answer_body = str(BeautifulSoup(answer_body, 'html5lib'))
        parsed_answer_body = document_fromstring(answer_body)
    for math_element in parsed_answer_body.xpath('//span[@class = "math-container"]'):
        math_tokens = math_element.text
        math_element.text = f'${math_tokens}$'
    answer_body_text = parsed_answer_body.text_content()
    answer_body_text = answer_body_text.strip()
    return answer_body_text


def read_task1_qrel_file(file_path):
    """
    Reading input file with relevance judgements for ARQMath-3 Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, and relevance judgements
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, _, document_id, relevance_judgement = row
            relevance_judgement = int(relevance_judgement)
            yield ((topic_id, document_id), relevance_judgement)


def get_relevant_task1_answers(all_answers_directory, qrel_file, data_reader_record):
    """
    Reading all relevant answers for ARQMath-3 Task 1 topics
    @param all_answers_directory: Input directory with all runs for ARQMath-3 Task 1
    @param qrel_file: Input file with relevance judgements for ARQMath-3 Task 1
    @param data_reader_record: ARQMathCode data reader
    @return: iterable of topic ids and answer body texts in text + LaTeX format
    """
    qrel_dict = dict(read_task1_qrel_file(qrel_file))

    all_answers_directory = Path(all_answers_directory)
    all_result_files = sorted(all_answers_directory.glob('*.tsv'))
    if not all_result_files:
        raise ValueError(f'No Task 1 result files found in directory {all_answers_directory}')

    relevant_answer_ids = set()
    for result_file in all_result_files:
        for topic_id, answer_id, *_ in read_task1_result_file(result_file):
            try:
                judgement = qrel_dict[topic_id, answer_id]
                if judgement > 1:
                    relevant_answer_ids.add((topic_id, answer_id))
            except KeyError:
                pass

    for topic_id, answer_id in relevant_answer_ids:
        answer_body_text = convert_task1_answer_id_to_answer(answer_id, data_reader_record)
        yield (topic_id, answer_body_text)


def read_task3_result_file(file_path):
    """
    Reading input results file in ARQMath format for ARQMath-3 Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids and answer body texts in text + LaTeX format
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, *_, answer_body_text = row
            yield topic_id, answer_body_text


def read_task3_map_file(file_path, expected_run_name):
    """
    Reading map file from topic IDs and run names to document IDs for ARQMath-3 Task 3
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


def read_task3_qrel_file(file_path):
    """
    Reading input file with relevance judgements for ARQMath-3 Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, and relevance judgements
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, _, answer_id, relevance_judgement = row
            relevance_judgement = int(relevance_judgement)
            yield ((topic_id, answer_id), relevance_judgement)


def get_relevant_task3_answers(all_answers_directory, qrel_file, map_file):
    """
    Reading all relevant answers for ARQMath-3 Task 1 topics
    @param all_answers_directory: Input directory with all runs for ARQMath-3 Task 1
    @param qrel_file: Input file with complete relevance judgements for ARQMath-3 Task 1
    @param map_file: Input map file from topic IDs and run names to synthetic answer IDs for ARQMath-3 Task 3
    @return: iterable of topic ids and answer body texts in text + LaTeX format
    """
    qrel_dict = dict(read_task3_qrel_file(qrel_file))

    all_answers_directory = Path(all_answers_directory)
    all_result_files = sorted(all_answers_directory.glob('*.tsv'))
    if not all_result_files:
        raise ValueError(f'No Task 3 result files found in directory {all_answers_directory}')

    for result_file in all_result_files:
        run_name = Path(result_file).stem
        map_dict = dict(read_task3_map_file(map_file, run_name))
        result_set = set()
        for topic_id, answer_body_text in read_task3_result_file(result_file):
            try:
                if topic_id in result_set:
                    raise ValueError(f'Repeated topic {topic_id} in {result_file}')
                result_set.add(topic_id)

                answer_id = map_dict[topic_id]
                judgement = qrel_dict[topic_id, answer_id]
                if judgement > 1 and judgement < 4:
                    yield topic_id, answer_body_text
            except KeyError:
                pass


def main():
    """
    example: pip install lxml beautifulsoup4 git+https://github.com/MIR-MU/ARQMathCode.git
             python3 evaluate_task3_results_automatic.py
               -all_task1_answers "task1_arqmath3_runs/"
               -all_task3_answers "task3_arqmath3_runs_without_GPT3/"
               -collection "collection/"
               -in "Baseline2022-task3-GPT3-auto-both-generate-P.tsv"
               -map "teams_answer_id.tsv"
               -task1_qrel "qrel_task1_2022_official.tsv"
               -task3_qrel "qrel_task3_2022_official_complete.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Compute Task 3 automatic evaluation measures (LS, CS) for Task 3 results')
    parser.add_argument('-all_task1_answers',
                        help=('Input directory with all runs for ARQMath-3 Task 1 except runs '
                              'from the same team whose run we are evaluating'),
                        required=True)
    parser.add_argument('-all_task3_answers',
                        help=('Input directory with all runs for ARQMath-3 Task 3 except runs '
                              'from the same team whose run we are evaluating'),
                        required=True)
    parser.add_argument('-collection',
                        help='Input directory with ARQMath-3 collection',
                        required=True)
    parser.add_argument('-in',
                        help='Input result file in ARQMath format for ARQMath-3 Task 3',
                        required=True)
    parser.add_argument('-map',
                        help=('Input map file from topic IDs and run names to synthetic '
                              'answer IDs for ARQMath-3 Task 3'),
                        required=True)
    parser.add_argument('-task1_qrel',
                        help='Input file with relevance judgements for ARQMath-3 Task 1',
                        required=True)
    parser.add_argument('-task3_qrel',
                        help=('Input file with complete relevance judgements (including 5: system '
                              'failure and 6: do not know judgements) for ARQMath-3 Task 3'),
                        required=True)

    args = vars(parser.parse_args())
    all_task1_answers_directory = args['all_task1_answers']
    all_task3_answers_directory = args['all_task3_answers']
    collection_directory = args['collection']
    result_file = args['in']
    map_file = args['map']
    task1_qrel_file = args['task1_qrel']
    task3_qrel_file = args['task3_qrel']

    data_reader_record = DataReaderRecord(collection_directory)

    relevant_task1_answers = get_relevant_task1_answers(
            all_task1_answers_directory, task1_qrel_file, data_reader_record)
    relevant_task3_answers = get_relevant_task3_answers(
            all_task3_answers_directory, task3_qrel_file, map_file)

    relevant_answers = defaultdict(lambda: set())
    for topic_id, answer in zip(relevant_task1_answers, relevant_task3_answers):
        relevant_answers[topic_id].add(answer)

    result_task3_answers = read_task3_result_file(result_file)  # noqa: F841


if __name__ == "__main__":
    main()
