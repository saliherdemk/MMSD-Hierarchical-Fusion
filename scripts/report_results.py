"""Check saved predictions and regenerate the report tables (standard library only)."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tables'
FAMILIES = {
    'Text-only': 'text_only',
    'Concat2, fine-tuned': 'concat2',
    'Concat3, ImageNet': 'concat3_base_model',
    'Concat3, fine-tuned': 'concat3_paper',
    'HFM, COCO attributes': 'fusion_full',
    'HFM, CLIP attributes': 'fusion_clip/results/runs/hierarchical_finetuned_clip_sentiment_packed',
    'Filtered COCO attributes': 'fusion_full_filtered',
    'Attribute dropout': 'fusion_attribute_dropout',
    'Gated early fusion': 'fusion_gated',
    'HFM, descriptive-only CLIP': 'fusion_clip/results_ablation/descriptive_only',
}


def metrics(rows):
    pairs = [(int(row['label']), int(row['prediction'])) for row in rows]
    if not pairs or any(y not in (0, 1) or p not in (0, 1) for y, p in pairs):
        raise ValueError('Expected nonempty binary labels and predictions')
    tp, fp, tn, fn = (pairs.count(pair) for pair in ((1, 1), (0, 1), (0, 0), (1, 0)))
    return {
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        'accuracy': (tp + tn) / len(pairs),
    }


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def table(filename, columns, heading, rows):
    # Match the paper's spacing style without changing numeric signs or math.
    def wording(text):
        return re.sub(r'(?<=\w)-+(?=\w)', ' ', text)

    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule', wording(heading) + r' \\', r'\midrule']
    lines.extend(' & '.join(wording(cell) for cell in row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (OUT / filename).write_text('\n'.join(lines) + '\n')


def check_validation(path, expected, reference_labels, hashes):
    rows = list(csv.DictReader(path.open()))
    labels = {row['tweet_id']: int(row['label']) for row in rows}
    assert len(rows) == len(labels) == 2410
    if reference_labels is not None:
        assert labels == reference_labels
    for row in rows:
        probability = float(row['probability'])
        assert math.isfinite(probability) and 0 <= probability <= 1
        assert int(row['prediction']) == int(probability > 0.5)
    for key, value in metrics(rows).items():
        close(value, expected[key])
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return labels


def main():
    close(metrics([{'label': y, 'prediction': p} for y, p in
                   [(1, 1), (1, 0), (0, 1), (0, 0)]])['f1'], 0.5)
    close(metrics([{'label': 0, 'prediction': 0}])['f1'], 0.0)
    OUT.mkdir(exist_ok=True)
    predictions, results, summaries, hashes = {}, {}, {}, {}
    reference_labels = None
    validation_labels = None
    validation_checked = 0
    checked = 0
    for name, folder in FAMILIES.items():
        path = ROOT / folder
        summary = json.loads((path / 'summary.json').read_text())
        summaries[name] = summary
        results[name] = []
        for seed in (42, 43, 44):
            run = path / f'seed_{seed}'
            result = json.loads((run / 'results.json').read_text())
            assert result['seed'] == seed
            history = json.loads((run / 'history.json').read_text())
            best = max(history, key=lambda record: record['validation']['f1'])
            close(best['validation']['f1'], result['best_validation_f1'])
            assert best['epoch'] == result['selected_epoch']
            if (run / 'validation_predictions.csv').exists():
                validation_labels = check_validation(
                    run / 'validation_predictions.csv', best['validation'], validation_labels, hashes)
                validation_checked += 1
            for file in sorted(run.glob('test_predictions*.csv')):
                rows = list(csv.DictReader(file.open()))
                assert len(rows) == 2409
                labels = {row['tweet_id']: int(row['label']) for row in rows}
                assert len(labels) == len(rows)
                if reference_labels is None:
                    reference_labels = labels
                assert labels == reference_labels
                for row in rows:
                    p = float(row['probability'])
                    assert math.isfinite(p) and 0 <= p <= 1
                    assert int(row['prediction']) == int(p > 0.5)
                condition = file.stem.removeprefix('test_predictions').lstrip('_')
                expected = result['conditions'][condition]['test'] if condition else result['test']
                for key, value in metrics(rows).items():
                    close(value, expected[key])
                if not condition:
                    predictions[name, seed] = {row['tweet_id']: row for row in rows}
                hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
                checked += 1
            results[name].append(result)
        for metric in ('precision', 'recall', 'f1', 'accuracy'):
            values = [row['test'][metric] for row in results[name]]
            close(mean(values), summary[metric]['mean'])
            close(stdev(values), summary[metric]['std'])

    def percentage(name, metric):
        value = summaries[name][metric]
        return f"${100 * value['mean']:.2f}\\pm{100 * value['std']:.2f}$"

    main_names = list(FAMILIES)[:6]
    table('local_results.tex', 'lrrrr',
          r'Model & Precision & Recall & F1 & Accuracy',
          [[name] + [percentage(name, key) for key in ('precision', 'recall', 'f1', 'accuracy')]
           for name in main_names])
    table('seed_results.tex', 'lrrr', 'Model & 42 & 43 & 44',
          [[name] + [f"{100 * row['test']['f1']:.2f}" for row in results[name]]
           for name in main_names])
    table('exploratory.tex', 'lrr', r'Attribute intervention & Test F1 & $\Delta$ F1 (pp)',
          [[name, percentage(name, 'f1'),
            f"{100 * (summaries[name]['f1']['mean'] - summaries['HFM, COCO attributes']['f1']['mean']):+.2f}"]
           for name in [*list(FAMILIES)[6:9], 'HFM, CLIP attributes']])
    resource_rows = []
    for name in main_names:
        records = results[name]
        training = mean(r.get('training_seconds', r.get('training_seconds_including_validation')) for r in records)
        testing = mean(r.get('test_seconds', r.get('test_seconds_including_data_loading')) for r in records)
        resource_rows.append([name, f'{training:,.2f}', f'{testing:.2f}'])
    table('resources.tex', 'lrr', r'Model & Train (s) & Test (s)', resource_rows)
    pairs = []
    for seed in (42, 43, 44):
        baseline = predictions['HFM, COCO attributes', seed]
        clip = predictions['HFM, CLIP attributes', seed]
        corrected = sum(baseline[k]['prediction'] != baseline[k]['label'] and
                        clip[k]['prediction'] == clip[k]['label'] for k in baseline)
        introduced = sum(baseline[k]['prediction'] == baseline[k]['label'] and
                         clip[k]['prediction'] != clip[k]['label'] for k in baseline)
        delta = metrics(list(clip.values()))['f1'] - metrics(list(baseline.values()))['f1']
        pairs.append([str(seed), str(corrected), str(introduced), f'{delta * 100:+.2f}'])
    table('paired_errors.tex', 'rrrr', r'Seed & Corrected & Introduced & $\Delta$ F1 (pp)', pairs)

    clip_names = ['HFM, descriptive-only CLIP', 'HFM, CLIP attributes']
    table('clip_ablation.tex', 'lrrrr', r'Attribute rule & Precision & Recall & F1 & Accuracy',
          [[label] + [percentage(name, key) for key in ('precision', 'recall', 'f1', 'accuracy')]
           for label, name in zip(['Descriptive-only', r'Sentiment-aware ($\tau=0$)'], clip_names)])
    table('clip_ablation_seeds.tex', 'rrrr',
          r'Seed & Descriptive-only & Sentiment-aware & $\Delta$ F1 (pp)',
          [[str(seed), *[f"{100 * results[name][i]['test']['f1']:.2f}" for name in clip_names],
            f"{100 * (results[clip_names[1]][i]['test']['f1'] - results[clip_names[0]][i]['test']['f1']):+.2f}"]
           for i, seed in enumerate((42, 43, 44))])
    screening = ROOT / 'fusion_clip/results_ablation/threshold_screening'
    selection = json.loads((screening / 'selection.json').read_text())
    assert [row['threshold'] for row in selection['screening']] == [0, 0.01, 0.02, 0.04]
    for row in selection['screening']:
        run = screening / 'seed_42' / row['variant']
        result = json.loads((run / 'validation_result.json').read_text())
        history = json.loads((run / 'history.json').read_text())
        best = max(history, key=lambda item: item['validation']['f1'])
        assert result['seed'] == row['seed'] == 42
        assert best['epoch'] == result['selected_epoch'] == row['selected_epoch']
        close(best['validation']['f1'], row['validation_f1'])
        for metric, value in best['validation'].items():
            close(value, result['validation'][metric])
        validation_labels = check_validation(
            run / 'validation_predictions.csv', result['validation'], validation_labels, hashes)
        validation_checked += 1
    winner = min(selection['screening'], key=lambda row: (-row['validation_f1'], row['threshold']))
    assert winner['threshold'] == selection['selected_threshold']
    assert winner['variant'] == selection['selected_variant']
    table('clip_thresholds.tex', 'rrr', r'Margin $\tau$ & Validation F1 & Affect insertion (\%)',
          [[f"{row['threshold']:g}", f"{100 * row['validation_f1']:.2f}",
            f"{100 * row['validation_insertion_rate']:.2f}"] for row in selection['screening']])
    audit = {'checked_prediction_files': checked, 'rows_per_file': 2409,
             'checked_validation_files': validation_checked, 'validation_rows_per_file': 2410,
             'checks': 'labels, IDs, threshold, metrics, summaries, validation checkpoint selection',
             'prediction_sha256': hashes}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Verified {checked} test and {validation_checked} validation prediction files '
          f'and {len(FAMILIES)} experiment summaries; wrote {OUT}')


if __name__ == '__main__':
    main()
