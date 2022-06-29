from collections import defaultdict
import csv
import logging

from arqmathcode.post_reader_record import DataReaderRecord
from lxml import etree
from lxml.html import document_fromstring
from bs4 import BeautifulSoup


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


def convert_task1_answer_id_to_answer(answer_id, data_reader_record):
    """
    Converting answer ids to answer bodies in text + LaTeX format
    @param answer_id: id of the answer
    @param data_reader_record: ARQMathCode data reader
    @return the body text of the answer in text + LaTeX format
    """
    answer = documents_text_reader.post_parser.map_just_answers[answer_id]
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


def read_task3_qrel_file(file_path):
    """
    Reading input file with relevance judgements for ARQMath Task 3
    @param file_path: file path to input file
    @return: iterable of topic ids, answer ids, and relevance judgements
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, _, answer_id, relevance_judgement = row
            relevance_judgement = int(relevance_judgement)
            yield ((topic_id, answer_id), relevance_judgement)


# pip install lxml beautifulsoup4 git+https://github.com/MIR-MU/ARQMathCode.git
# from arqmathcode.post_reader_record import DataReaderRecord
# data_reader_record = DataReaderRecord(str(documents_text))
