import argparse
from collections import defaultdict
import csv
import logging
from pathlib import Path
from statistics import mean
import re

from arqmathcode.post_reader_record import DataReaderRecord
from lxml import etree
from lxml.html import document_fromstring
from bs4 import BeautifulSoup
from transformers import AutoTokenizer


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
        math_element.text = f' [MATH] {math_tokens} [/MATH] '
    answer_body_text = parsed_answer_body.text_content()
    answer_body_text = answer_body_text.rstrip()
    if not answer_body_text.startswith(' [MATH]'):
        answer_body_text = answer_body_text.lstrip()
    answer_body_text = re.sub(r' \[/MATH\] \[MATH\] ', ' ', answer_body_text)
    answer_body_text = re.sub(r' \[/MATH\]$', '', answer_body_text)
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
                if judgement not in {0, 1, 2, 3, 5, 6}:
                    raise ValueError(f'Unknown judgement value {judgement}, expected 0-3, 5, or 6')
                if judgement > 1 and judgement < 4:
                    yield topic_id, answer_body_text
            except KeyError:
                pass


def write_all_relevant_answers(all_relevant_answers, file_path):
    """
    Writing relevant answers to an output file
    @param all_relevant_answers: dict of topic_ids and lists of answer body texts in text + LaTeX format
    @param file_path: file path to output file
    @return:
    """
    csv_writer = csv.writer(file_path, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for topic_id, relevant_answers in sorted(all_relevant_answers):
        for relevant_answer in relevant_answers:
            row = (topic_id, relevant_answer)
            csv_writer.writerow(row)


def compute_contextual_similarity(answer, relevant_answers):
    """
    Computing contextual similarity between an answer and the most similar out of relevant answers
    @param answer: answer body text in text + LaTeX format
    @param relevant_answers: set of answer body texts in text + LaTeX format
    @return: contextual similarity
    """
    return 1.0  # TODO


def compute_lexical_overlap(answer, relevant_answers):
    """
    Computing lexical overlap between an answer and the most similar out of relevant answers
    @param answer: answer body text in text + LaTeX format
    @param relevant_answers: set of answer body texts in text + LaTeX format
    @return: lexical overlap
    """
    return 1.0  # TODO


def main():
    """
    example: pip install lxml beautifulsoup4 transformers>=4.20.0 git+https://github.com/MIR-MU/ARQMathCode.git
             python3 evaluate_task3_results_automatic.py
               -all_task1_answers "task1_arqmath3_runs/"
               -all_task3_answers "task3_arqmath3_runs_without_GPT3/"
               -collection "collection/"
               -in "Baseline2022-task3-GPT3-auto-both-generate-P.tsv"
               -map "teams_answer_id.tsv"
               -task1_qrel "qrel_task1_2022_official.tsv"
               -task3_qrel "qrel_task3_2022_official_complete.tsv"
               -relevant_answer_dump "relevant_answers.tsv"
    @return:
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Compute Task 3 automatic evaluation measures (LO, CS) for Task 3 results')
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
    parser.add_argument('-relevant_answer_dump',
                        help=('Output file with all H+M anwers for ARQMath-3 Task 1 and ARQMath-3 '
                              'Task 3 in TSV format with topic ids and relevant answer texts. '
                              'Optional, but useful for debugging.'),
                        required=False)

    args = vars(parser.parse_args())
    all_task1_answers_directory = args['all_task1_answers']
    all_task3_answers_directory = args['all_task3_answers']
    collection_directory = args['collection']
    result_file = args['in']
    map_file = args['map']
    task1_qrel_file = args['task1_qrel']
    task3_qrel_file = args['task3_qrel']
    output_all_relevant_answers_file = args['relevant_answer_dump']

    data_reader_record = DataReaderRecord(collection_directory)

    all_relevant_task1_answers = get_relevant_task1_answers(
            all_task1_answers_directory, task1_qrel_file, data_reader_record)
    all_relevant_task3_answers = get_relevant_task3_answers(
            all_task3_answers_directory, task3_qrel_file, map_file)

    all_relevant_answers = defaultdict(lambda: set())
    for topic_id, answer in zip(all_relevant_task1_answers, all_relevant_task3_answers):
        all_relevant_answers[topic_id].add(answer)

    if output_all_relevant_answers_file is not None:
        write_all_relevant_answers(all_relevant_answers, output_all_relevant_answers_file)

    result_answers = []
    missing_topics = set()
    for topic_id, answer in read_task3_result_file(result_file):
        try:
            relevant_answers = all_relevant_answers[topic_id]
            result_answers.append((answer, relevant_answers))
        except KeyError:
            missing_topics.add(topic_id)

    if missing_topics:
        LOGGER.warning(f'Results for {len(missing_topics)} topics had no relevant answers: {sorted(missing_topics)}')
        LOGGER.warning(f'Running the evaluation using just {len(result_answers)} topics')

    tokenizer = AutoTokenizer.from_pretrained('witiko/mathberta', add_prefix_space=True)  # noqa

    lexical_overlaps = []
    contextual_similarities = []
    for answer, relevant_answers in result_answers:
        partial_lexical_overlap = compute_lexical_overlap(answer, relevant_answers)
        partial_contextual_similarity = compute_contextual_similarity(answer, relevant_answers)
        lexical_overlaps.append(partial_lexical_overlap)
        contextual_similarities.append(partial_contextual_similarity)

    lexical_overlap = mean(lexical_overlaps)
    contextual_similarity = mean(contextual_similarities)

    print(f'LO: {lexical_overlap:.3f}')
    print(f'CS: {contextual_similarity:.3f}')


if __name__ == "__main__":
    main()
