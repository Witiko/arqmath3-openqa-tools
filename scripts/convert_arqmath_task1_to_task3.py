from pv211_utils.arqmath.loader import load_answers

import argparse
from collections import defaultdict
import csv


def read_task1_result_file(file_path):
    """
    Reading input file in ARQMath format for ARQMath Task 1
    @param file_path: file path to input file
    @return: dict of topic ids and results. results is dict of answer ids, ranks, scores, and run identifiers
    """
    result = defaultdict(lambda: dict())
    with open(file_path, 'rt', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            topic_id, answer_id, rank, score, run_id = row
            result[topic_id][answer_id] = (int(rank), float(score), run_id)
    return result


def write_task3_result_file(file_path, result, max_length=1200):
    """
    Reading input file in ARQMath format for ARQMath Task 3
    @param file_path: file path to output file
    @param result: dict of topic ids and results. results is dict of answer ids, ranks, scores, and run identifiers
    @param max_length: maximum number of unicode characters in an answer for ARQMath Task 3
    @return:
    """
    answers = load_answers(text_format='text+latex')
    result_file = open(file_path, 'wt', newline='', encoding='utf-8')
    csv_writer = csv.writer(result_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for topic_id in sorted(result):
        rank_getter = lambda item: item[1][0]  # noqa:E731
        retrieval_results = sorted(result[topic_id].items(), key=rank_getter)
        for answer_id, (rank, score, run_id) in retrieval_results:
            if answer_id not in answers:
                print('Warning: Text of answer with ID {} not found in pv211-utils'.format(answer_id))
                continue
            answer = answers[answer_id].body[:max_length]
            row = (topic_id, str(rank), str(score), run_id, answer)
            csv_writer.writerow(row)
            break  # Only use the topmost result as the answer for ARQMath Task 3


def main():
    """
    example: pip install git+https://github.com/MIR-MU/pv211-utils.git
             python3 convert_arqmath_task1_to_task3.py -in "TangentS_task1_2021.tsv" -out "TangentS_task3_2021.tsv"
    @return:
    """
    parser = argparse.ArgumentParser(description='Convert results from ARQMath Task 1 to Task 3')

    parser.add_argument('-in', help='Input result file in ARQMath format for ARQMath Task 1', required=True)
    parser.add_argument('-out', help='Output result file in ARQMath format for ARQMath Task 3', required=True)

    args = vars(parser.parse_args())
    input_file = args['in']
    output_file = args['out']

    results = read_task1_result_file(input_file)
    write_task3_result_file(output_file, results)


if __name__ == "__main__":
    main()
