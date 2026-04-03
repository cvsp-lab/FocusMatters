import os
import json
import argparse
from tqdm import tqdm


def evaluate_pope(ans_file, label_file):
    """
    Evaluate POPE metrics from answer and label files.

    Args:
        ans_file: path to the POPE answer JSONL file
        label_file: path to the POPE label JSONL file

    Returns:
        dict with keys: Accuracy, Precision, Recall, F1, Yes_ratio,
                        TP, FP, TN, FN, no_answer
    """
    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    # Parse answers
    for idx, answer in enumerate(answers):
        text = answer['answer']

        if text is None:
            answer['answer'] = 'wrong'
        else:
            if text.find('.') != -1:
                text = text.split('.')[0]

            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                answer['answer'] = 'no'
            else:
                answer['answer'] = 'yes'

    # Convert labels to binary
    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    # Convert predictions to numeric
    pred_list = []
    for answer in answers:
        if answer['answer'] == 'wrong':
            pred_list.append(-1)
        elif answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0.0

    TP, TN, FP, FN, no_answer = 0, 0, 0, 0, 0
    for pred, label in tqdm(zip(pred_list, label_list), total=len(pred_list)):
        if pred == -1:
            no_answer += 1
        elif pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN + no_answer) if (TP + TN + FP + FN + no_answer) > 0 else 0.0

    metrics = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Yes_ratio': yes_ratio,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'no_answer': no_answer,
    }

    return metrics


def print_metrics(all_pope_metrics, save_result_file, quiet=False):
    """
    Print POPE metrics for all targets and save them to a JSON file.

    Args:
        all_pope_metrics: dict of {pope_type: metrics_dict}
                          e.g. {'random': {...}, 'popular': {...}, 'adversarial': {...}}
        save_result_file: path to the JSON result file
        quiet: if True, suppress printing
    """
    if not quiet:
        for pope_type, metrics in all_pope_metrics.items():
            print(f'\n=== POPE [{pope_type}] ===')
            print('TP\tFP\tTN\tFN\tNOANSWER')
            print('{}\t{}\t{}\t{}\t{}'.format(
                metrics['TP'], metrics['FP'], metrics['TN'],
                metrics['FN'], metrics['no_answer']))
            for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'Yes_ratio']:
                k_str = str(k).ljust(12)
                v_str = f'{metrics[k]:.4f}'
                print(k_str, v_str, sep=': ')

    # pope_type별로 키에 접미사를 붙여 하나의 JSON에 저장
    # e.g. Accuracy_random, F1_popular, ...
    pope_data = {}
    for pope_type, metrics in all_pope_metrics.items():
        for k, v in metrics.items():
            pope_data[f'{k}_{pope_type}'] = v

    dir_name = os.path.dirname(save_result_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(save_result_file, 'w') as f:
        json.dump(pope_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POPE evaluation with JSON metric recording')
    parser.add_argument('--ans_file', type=str, nargs='+', required=True,
                        help='Prefix(es) for answer file(s) (without .json extension). '
                             'One per POPE target. e.g. result_path/output_random result_path/output_popular')
    parser.add_argument('--pope_type', type=str, nargs='+', required=True,
                        help='POPE target type(s) corresponding to each ans_file. '
                             'e.g. random popular adversarial')
    parser.add_argument('--pope_result_file', type=str, required=True,
                        help='Path to JSON file to save POPE results')
    parser.add_argument('--model', type=str, default='llava')
    args = parser.parse_args()

    assert len(args.ans_file) == len(args.pope_type), \
        f'Number of ans_files ({len(args.ans_file)}) must match number of pope_types ({len(args.pope_type)})'

    all_pope_metrics = {}
    for ans_prefix, pope_type in zip(args.ans_file, args.pope_type):
        ans_file = f'{ans_prefix}.json'
        label_file = f'{ans_prefix}_label.json'
        print(f'\n[POPE] Evaluating {pope_type}: {ans_file}')
        metrics = evaluate_pope(ans_file, label_file)
        all_pope_metrics[pope_type] = metrics

    print_metrics(all_pope_metrics, args.pope_result_file)
    print(f'\n[POPE] Saved all metrics to: {args.pope_result_file}')
