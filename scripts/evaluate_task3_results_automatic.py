import argparse
from collections import defaultdict
from functools import lru_cache
import csv
import json
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
    @return: iterable of topic ids, answer ids, and run ids
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, answer_id, *_, run_id = row
            yield topic_id, answer_id, run_id


def normalize_answer_text(answer):
    """
    Normalizing the body text of an answer in text + LaTeX format
    @param answer: answer body text in text + LaTeX format
    @return: the normalized answer body text in text + LaTeX format
    """
    answer = answer.rstrip()
    if not answer.startswith(' [MATH]'):
        answer = answer.lstrip()
    answer = re.sub(r' \[/MATH\] \[MATH\] ', ' ', answer)
    answer = re.sub(r' \[/MATH\]$', '', answer)
    return answer


def convert_task1_answer_id_to_answer(answer_id, data_reader_record, max_answer_length=1200):
    """
    Converting answer ids to answer bodies in text + LaTeX format
    @param answer_id: id of the answer
    @param data_reader_record: ARQMathCode data reader
    @param max_answer_length: the maximum length of an answer
    @return: the body text of the answer in text + LaTeX format or None if the answer was too long
    """
    answer = data_reader_record.post_parser.map_just_answers[answer_id]
    answer_body = answer.body
    try:
        parsed_answer_body = document_fromstring(answer_body)
    except etree.ParserError:
        answer_body = str(BeautifulSoup(answer_body, 'html5lib'))
        parsed_answer_body = document_fromstring(answer_body)
    math_elements = parsed_answer_body.xpath('//span[@class = "math-container"]')
    for math_element in math_elements:
        math_tokens = math_element.text
        math_element.text = f' [MATH] {math_tokens} [/MATH] '
    answer_body_text = parsed_answer_body.text_content()

    math_tag_length_overhead = len(math_elements) * (len(' [MATH] ') + len(' [/MATH] ') - len('$$'))
    answer_length = len(answer_body_text) - math_tag_length_overhead
    if answer_length > max_answer_length:
        return None

    answer_body_text = normalize_answer_text(answer_body_text)
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


def get_relevant_task1_answers(all_answers_directory, qrel_file, data_reader_record,
                               excluded_run_ids):
    """
    Reading all relevant answers for ARQMath-3 Task 1 topics
    @param all_answers_directory: input directory with all runs for ARQMath-3 Task 1
    @param qrel_file: input file with relevance judgements for ARQMath-3 Task 1
    @param data_reader_record: ARQMathCode data reader
    @param excluded_run_ids: set of blacklisted run ids that will disqualify answers uniquely contributed ny them
    @return: iterable of topic ids and answer body texts in text + LaTeX format
    """
    qrel_dict = dict(read_task1_qrel_file(qrel_file))

    all_answers_directory = Path(all_answers_directory)
    all_result_files = sorted(all_answers_directory.glob('*.tsv'))
    if not all_result_files:
        raise ValueError(f'No Task 1 result files found in directory {all_answers_directory}')

    relevant_answer_ids = set()
    answer_run_ids = defaultdict(lambda: set())
    for result_file in all_result_files:
        for topic_id, answer_id, run_id in read_task1_result_file(result_file):
            try:
                judgement = qrel_dict[topic_id, answer_id]
                if judgement > 1:
                    answer_run_ids[topic_id, answer_id].add(run_id)
                    relevant_answer_ids.add((topic_id, answer_id))
            except KeyError:
                pass

    for topic_id, answer_id in relevant_answer_ids:
        run_ids = answer_run_ids[topic_id, answer_id]
        if not (run_ids - excluded_run_ids):
            continue
        answer_body_text = convert_task1_answer_id_to_answer(answer_id, data_reader_record)
        if answer_body_text is not None:
            yield topic_id, answer_body_text


def replace_dollars_with_math_tags(answer):
    """
    Replacing dollar signs with [MATH] and [/MATH] tags in the body text of an answer in text + LaTeX format
    @param answer: answer body text in text + LaTeX format
    @return: the answer body text in text + LaTeX format after the replacement or None if malformed
    """
    dollar_regex = r'(?:^|[^\\])\$'

    if len(re.findall(dollar_regex, answer)) % 2 == 1:
        return None

    is_start_tag = True

    def replace_dollar_with_math_tag(match):
        nonlocal is_start_tag
        if is_start_tag:
            replacement = ' [MATH] '
        else:
            replacement = ' [/MATH] '
        is_start_tag = not is_start_tag
        return replacement

    answer = re.sub(dollar_regex, replace_dollar_with_math_tag, answer)
    return answer


def read_task3_result_file(file_path, max_answer_length=1200):
    """
    Reading input results file in ARQMath format for ARQMath-3 Task 3
    @param file_path: file path to input file
    @param max_answer_length: the maximum length of an answer
    @return: iterable of topic ids, run ids and answer body texts in text + LaTeX format
    """
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for line_number, row in enumerate(csv_reader):
            line_number += 1
            topic_id, _, __, run_id, ___, answer_body_text = row
            if len(answer_body_text) > max_answer_length:
                raise ValueError(f'Answer to topic {topic_id} on line {line_number} contains '
                                 f'{len(answer_body_text)} Unicode characters, but at most '
                                 f'{max_answer_length} were expected.')
            answer_body_text = replace_dollars_with_math_tags(answer_body_text)
            if answer_body_text is None:
                LOGGER.warning(f'Found an odd number of dollar signs ($) in answer to topic {topic_id} '
                               f'on line {line_number} from run file {file_path}. Skipping.')
                continue
            answer_body_text = normalize_answer_text(answer_body_text)
            yield topic_id, run_id, answer_body_text


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


def get_relevant_task3_answers(all_answers_directory, qrel_file, map_file, excluded_run_ids):
    """
    Reading all relevant answers for ARQMath-3 Task 1 topics
    @param all_answers_directory: Input directory with all runs for ARQMath-3 Task 1
    @param qrel_file: Input file with complete relevance judgements for ARQMath-3 Task 1
    @param map_file: Input map file from topic IDs and run names to synthetic answer IDs for ARQMath-3 Task 3
    @param excluded_run_ids: set of blacklisted run ids that will disqualify answers uniquely contributed ny them
    @return: iterable of topic ids and answer body texts in text + LaTeX format
    """
    qrel_dict = dict(read_task3_qrel_file(qrel_file))

    all_answers_directory = Path(all_answers_directory)
    all_result_files = sorted(all_answers_directory.glob('*.tsv'))
    if not all_result_files:
        raise ValueError(f'No Task 3 result files found in directory {all_answers_directory}')

    relevant_answers = set()
    answer_run_ids = defaultdict(lambda: set())
    for result_file in all_result_files:
        run_name = Path(result_file).stem
        map_dict = dict(read_task3_map_file(map_file, run_name))
        result_set = set()
        for topic_id, run_id, answer_body_text in read_task3_result_file(result_file):
            if topic_id in result_set:
                raise ValueError(f'Repeated topic {topic_id} in {result_file}')
            result_set.add(topic_id)
            try:
                answer_id = map_dict[topic_id]
                judgement = qrel_dict[topic_id, answer_id]
                if judgement not in {0, 1, 2, 3, 5, 6}:
                    raise ValueError(f'Unknown judgement value {judgement}, expected 0-3, 5, or 6')
                if judgement > 1 and judgement < 4:
                    answer_run_ids[topic_id, answer_body_text].add(run_id)
                    relevant_answers.add((topic_id, answer_body_text))
            except KeyError:
                pass

    for topic_id, answer_body_text in relevant_answers:
        run_ids = answer_run_ids[topic_id, answer_body_text]
        if not (run_ids - excluded_run_ids):
            continue
        yield topic_id, answer_body_text


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


@lru_cache(maxsize=None)
def tokenize(tokenizer, answer):
    """
    Tokenizing an answer body text in text + LaTeX format
    @param a tokenizer for answer body texts in text + LaTeX format
    @param answer: answer body text in text + LaTeX format
    @return: set of tokens
    """
    tokens = tokenizer.tokenize(answer)
    tokens = set(tokens)
    return tokens


def compute_contextual_similarity(topic_id, answer, relevant_answers):
    """
    Computing contextual similarity between an answer and the most similar out of relevant answers
    @param topic_id: the id of the topic for which the answer was given
    @param answer: answer body text in text + LaTeX format
    @param relevant_answers: set of relevant answer body texts in text + LaTeX format
    @return: contextual similarity
    """
    return 1.0  # TODO


def compute_lexical_overlap(topic_id, answer, relevant_answers, tokenizer):
    """
    Computing lexical overlap between an answer and the most similar out of relevant answers
    @param topic_id: the id of the topic for which the answer was given
    @param answer: answer body text in text + LaTeX format
    @param relevant_answers: set of relevant answer body texts in text + LaTeX format
    @param a tokenizer for answer body texts in text + LaTeX format
    @return: lexical overlap
    """
    tokenized_answer = tokenize(answer)
    tokenized_relevant_answers = map(tokenize, relevant_answers)
    tokenized_relevant_answers = filter(len, tokenized_relevant_answers)
    tokenized_relevant_answers = set(tokenized_relevant_answers)
    if not tokenized_relevant_answers:
        raise ValueError(f'No non-empty relevant answers for topic {topic_id}')

    best_f1_score = float('-inf')
    for tokenized_relevant_answer in tokenized_relevant_answers:
        intersection = tokenized_answer & tokenized_relevant_answer
        union = tokenized_answer | tokenized_relevant_answer
        precision = 1.0 * len(intersection) / len(tokenized_answer)
        recall = 1.0 * len(intersection) / len(tokenized_relevant_answer)
        f1_score = 2 * precision * recall / (precision + recall)
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    return best_f1_score


def main():
    """
    example: pip install lxml beautifulsoup4 transformers>=4.20.0 bert-score==0.3.11 git+https://github.com/MIR-MU/ARQMathCode.git
             python3 evaluate_task3_results_automatic.py
               -all_task1_answers "task1_arqmath3_runs/"
               -all_task3_answers "task3_arqmath3_runs/"
               -exclude_run_ids "['GPT3']"
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
                        help=('Input directory with all runs for ARQMath-3 Task 1'),
                        required=True)
    parser.add_argument('-all_task3_answers',
                        help=('Input directory with all runs for ARQMath-3 Task 3'),
                        required=True)
    parser.add_argument('-excluded_run_ids',
                        help=('A JSON array of run ids of results from the same team as the '
                              'result file being evaluated including the run id of the result '
                              'file being evaluated; all relevant answers contributed '
                              'uniquely by result files with these run ids will be excluded '
                              'from the evaluation'),
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
    excluded_run_ids = args['excluded_run_ids']
    collection_directory = args['collection']
    result_file = args['in']
    map_file = args['map']
    task1_qrel_file = args['task1_qrel']
    task3_qrel_file = args['task3_qrel']
    output_all_relevant_answers_file = args['relevant_answer_dump']

    excluded_run_ids_set = set(json.loads(excluded_run_ids))
    LOGGER.info(f'Excluded run ids: {sorted(excluded_run_ids_set)}')

    data_reader_record = DataReaderRecord(collection_directory)

    all_relevant_task1_answers = get_relevant_task1_answers(
            all_task1_answers_directory, task1_qrel_file, data_reader_record, excluded_run_ids_set)
    all_relevant_task3_answers = get_relevant_task3_answers(
            all_task3_answers_directory, task3_qrel_file, map_file, excluded_run_ids_set)

    all_relevant_answers = defaultdict(lambda: set())
    for topic_id, answer in zip(all_relevant_task1_answers, all_relevant_task3_answers):
        all_relevant_answers[topic_id].add(answer)

    if output_all_relevant_answers_file is not None:
        write_all_relevant_answers(all_relevant_answers, output_all_relevant_answers_file)

    result_answers = []
    missing_topics = set()
    for topic_id, _, answer in read_task3_result_file(result_file):
        try:
            relevant_answers = all_relevant_answers[topic_id]
            result_answers.append((topic_id, answer, relevant_answers))
        except KeyError:
            missing_topics.add(topic_id)

    if missing_topics:
        LOGGER.warning(f'Results for {len(missing_topics)} topics had no relevant answers: {sorted(missing_topics)}')
        LOGGER.warning(f'Running the evaluation using just {len(result_answers)} topics')

    tokenizer = AutoTokenizer.from_pretrained('witiko/mathberta', add_prefix_space=True)

    lexical_overlaps = []
    contextual_similarities = []
    for topic_id, answer, relevant_answers in tokenized_result_answers:
        partial_lexical_overlap = compute_lexical_overlap(topic_id, answer, relevant_answers, tokenizer)
        partial_contextual_similarity = compute_contextual_similarity(topic_id, answer, relevant_answers)
        lexical_overlaps.append(partial_lexical_overlap)
        contextual_similarities.append(partial_contextual_similarity)

    lexical_overlap = mean(lexical_overlaps)
    contextual_similarity = mean(contextual_similarities)

    print(f'LO: {lexical_overlap:.3f}')
    print(f'CS: {contextual_similarity:.3f}')


if __name__ == "__main__":
    main()
