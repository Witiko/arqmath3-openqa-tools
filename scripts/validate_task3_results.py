import argparse
from collections import defaultdict
import csv


def read_task3_result_file(file_path, max_answer_length=1200):
    """
    Reading input file in ARQMath format for ARQMath Task 3
    @param file_path: file path to input file
    @return: dict of topic ids and results
    """
    result = defaultdict(lambda: dict())
    with open(file_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for line_number, row in enumerate(csv_reader):
            topic_id, rank, score, run_id, answer = row
            if len(answer) > max_answer_length:
                raise ValueError(f'Answer to topic {topic_id} on line {line_number + 1} contains '
                                 f'{len(answer)} Unicode characters, but at most '
                                 f'{max_answer_length} were expected')
            result[topic_id] = (topic_id, str(rank), str(score), run_id, answer)
    return result


def main():
    """
    example: python3 validate_task3_results.py -in "TangentS_task3_2021.tsv"
    @return:
    """
    parser = argparse.ArgumentParser(description='Convert results from ARQMath Task 3')

    parser.add_argument('-in', help='Input result file in ARQMath format for ARQMath Task 3', required=True)

    args = vars(parser.parse_args())

    input_file = args['in']
    _ = read_task3_result_file(input_file)
    print(f'{input_file} validates')


if __name__ == "__main__":
    main()
