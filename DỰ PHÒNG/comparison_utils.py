from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def export_predictions(manifest, y_true, probabilities, output_dir, stem, label_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    probs = np.asarray(probabilities)
    truth = np.asarray(y_true)
    if probs.shape != (len(frame), len(label_names)) or not np.isfinite(probs).all():
        raise ValueError('Invalid prediction shape or non-finite probabilities')
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5) or (probs < 0).any():
        raise ValueError('Predictions must be normalized probabilities')
    if not np.array_equal(frame.label_id.to_numpy(), truth):
        raise ValueError('Prediction targets do not match manifest order')
    pred = probs.argmax(axis=1)
    result = frame[['image_id', 'source_row_index']].copy()
    result['y_true'], result['y_pred'] = truth, pred
    result['correct'] = pred == truth
    if 'soft_label' in frame:
        soft = np.stack([np.fromstring(str(v), sep=' ') for v in frame.soft_label])
        top = soft == soft.max(axis=1, keepdims=True)
        result['target_is_tie'] = top.sum(axis=1) > 1
        result['top_reference_classes'] = [' '.join(map(str, np.flatnonzero(row))) for row in top]
        result['matches_any_top_class'] = top[np.arange(len(pred)), pred]
        # correct and classification report use the fixed manifest reference label;
        # matches_any_top_class is a separate ambiguity-aware descriptive measure.
    for c, label in enumerate(label_names):
        result['prob_' + label.lower()] = probs[:, c]
    result.to_csv(output_dir / f'predictions_{stem}.csv', index=False)
    report = classification_report(truth, pred, labels=list(range(len(label_names))),
                                   target_names=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / f'classification_report_{stem}.csv', index_label='class')
    pd.DataFrame(confusion_matrix(truth, pred, labels=list(range(len(label_names)))),
                 index=label_names, columns=label_names).to_csv(output_dir / f'confusion_matrix_{stem}.csv', index_label='true_class')
    return result


def compare_paired_predictions(baseline, candidate, output_dir, stem='primary', alpha=0.05):
    """Two-sided exact McNemar on correctness, paired by ID (not row position).

    Overall is the primary test. Rare-class recall comparisons are exploratory,
    Holm-adjusted across the three classes. This does not test Macro-F1.
    """
    a = pd.read_csv(baseline) if not isinstance(baseline, pd.DataFrame) else baseline.copy()
    b = pd.read_csv(candidate) if not isinstance(candidate, pd.DataFrame) else candidate.copy()
    for frame in (a, b):
        if frame[['y_true', 'y_pred']].isna().any().any() or not frame[['y_true', 'y_pred']].isin(range(8)).all().all():
            raise ValueError('Labels must be integers in 0..7 without missing values')
        if frame.image_id.isna().any() or frame.image_id.duplicated().any():
            raise ValueError('Prediction IDs must be unique and non-null')
    if len(a) == 0 or set(a.image_id) != set(b.image_id):
        raise ValueError('Both models must predict exactly the same non-empty test set')
    paired = a[['image_id','y_true','y_pred']].merge(b[['image_id','y_true','y_pred']],
                 on='image_id', validate='one_to_one', suffixes=('_old','_new'))
    if not np.array_equal(paired.y_true_old, paired.y_true_new):
        raise ValueError('True labels differ between models')
    paired['old_correct'] = paired.y_pred_old == paired.y_true_old
    paired['new_correct'] = paired.y_pred_new == paired.y_true_new
    rows = []
    for name, subset in [('overall', paired), *[(label, paired[paired.y_true_old==c]) for c,label in [(5,'Disgust'),(6,'Fear'),(7,'Contempt')]]]:
        old, new = subset.old_correct.to_numpy(), subset.new_correct.to_numpy()
        both = int((old & new).sum()); lost = int((old & ~new).sum())
        gained = int((~old & new).sum()); neither = int((~old & ~new).sum())
        p = binomtest(lost, lost+gained, .5, alternative='two-sided').pvalue if lost+gained else 1.0
        rows.append(dict(scope=name, n=len(subset), both_correct=both, old_only_correct=lost,
            new_only_correct=gained, both_wrong=neither, discordant=lost+gained,
            p_exact=p if len(subset) else np.nan, alpha=alpha,
            accuracy_old=float(old.mean()) if len(old) else np.nan,
            accuracy_new=float(new.mean()) if len(new) else np.nan,
            delta_accuracy=float(new.mean()-old.mean()) if len(old) else np.nan))
    results = pd.DataFrame(rows)
    results['p_holm_rare'] = np.nan
    valid = results.index[(results.scope != 'overall') & results.p_exact.notna()].tolist()
    order = sorted(valid, key=lambda i: results.loc[i,'p_exact'])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order)-rank)*results.loc[i,'p_exact']))
        results.loc[i,'p_holm_rare'] = running
    results['significant'] = False
    results.loc[0,'significant'] = results.loc[0,'p_exact'] < alpha
    results.loc[valid,'significant'] = results.loc[valid,'p_holm_rare'] < alpha
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / f'paired_predictions_{stem}.csv', index=False)
    results.to_csv(out / f'mcnemar_{stem}.csv', index=False)
    r = rows[0]
    pd.DataFrame([[r['both_correct'],r['old_only_correct']],[r['new_only_correct'],r['both_wrong']]],
                 index=['old_correct','old_wrong'], columns=['new_correct','new_wrong']).to_csv(out / f'mcnemar_table_{stem}.csv')
    metrics = []
    for label, pred in [('baseline',paired.y_pred_old),('candidate',paired.y_pred_new)]:
        truth = paired.y_true_old
        acc = accuracy_score(truth,pred); mf1 = f1_score(truth,pred,labels=range(8),average='macro',zero_division=0)
        metrics.append(dict(model=label, accuracy=acc, macro_f1=mf1,
            weighted_f1=f1_score(truth,pred,labels=range(8),average='weighted',zero_division=0), composite=.5*(acc+mf1)))
    summary = pd.DataFrame(metrics).set_index('model')
    summary.loc['delta_new_minus_old'] = summary.loc['candidate']-summary.loc['baseline']
    summary.to_csv(out / f'comparison_vs_baseline_{stem}.csv')
    return results


def compare_test_intersection(drop_dir, keep_dir, output_dir, label_names, protocol='single'):
    """Compare separately selected models ONLY on their common held-out images.

    image_id is the SHA-256 of canonical raw pixels, not a row position.
    Own-test reports remain separate. No training, threshold tuning or resplitting.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    branches = []

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    for name, folder in [('drop_ties', drop_dir), ('keep_ties', keep_dir)]:
        folder = Path(folder)
        manifest_dir = folder / 'split_manifest'
        config = json.loads((manifest_dir / 'split_config.json').read_text(encoding='utf-8'))
        experiment = json.loads((folder / 'experiment_config.json').read_text(encoding='utf-8'))
        meta = json.loads((folder / 'evaluation_metadata.json').read_text(encoding='utf-8'))
        prediction_path = folder / f'predictions_selected_test_{protocol}.csv'
        if meta['branch'] != name or meta['protocol'] != protocol or experiment['evaluation_protocol'] != protocol:
            raise ValueError('Both branches must use the same predeclared prediction protocol')
        for field, file in [('bundle_sha256', folder / 'best_bundle.pth'),
                            ('test_manifest_sha256', manifest_dir / 'test_ids.csv'),
                            ('predictions_sha256', prediction_path)]:
            if meta[field] != sha(file):
                raise ValueError(f'{name}: stale checkpoint/manifest/predictions; rerun evaluation')
        if config['parameters']['drop_ties'] != (name == 'drop_ties'):
            raise ValueError(f'{name}: incorrect tie policy')
        manifests = {s: pd.read_csv(manifest_dir / f'{s}_ids.csv') for s in ('train', 'val', 'test')}
        all_rows = pd.concat(manifests.values(), ignore_index=True)
        if all_rows.image_id.isna().any() or all_rows.image_id.duplicated().any():
            raise ValueError(f'{name}: duplicate/missing IDs or overlap between its own splits')
        predictions = pd.read_csv(prediction_path)
        if predictions.image_id.isna().any() or predictions.image_id.duplicated().any():
            raise ValueError(f'{name}: duplicate/missing prediction IDs')
        test = manifests['test']
        if set(predictions.image_id) != set(test.image_id):
            raise ValueError(f'{name}: predictions do not cover exactly its test manifest')
        aligned = predictions.set_index('image_id').loc[test.image_id]
        if not np.array_equal(aligned.y_true.to_numpy(), test.label_id.to_numpy()):
            raise ValueError(f'{name}: prediction truth differs from manifest')
        if not np.array_equal(aligned.source_row_index.to_numpy(), test.source_row_index.to_numpy()):
            raise ValueError(f'{name}: prediction source rows differ from manifest')
        branches.append(dict(name=name, config=config, manifests=manifests, predictions=predictions,
                             dedup=pd.read_csv(manifest_dir / 'dedup_report.csv')))

    a, b = branches
    if a['config']['source_files_sha256'] != b['config']['source_files_sha256']:
        raise ValueError('Branches must originate from the same source image/label CSVs')
    if a['config']['preprocessing_config'] != b['config']['preprocessing_config']:
        raise ValueError('Preprocessing differs between branches')
    common = set(a['manifests']['test'].image_id) & set(b['manifests']['test'].image_id)
    # Union duplicate relationships recorded by BOTH independent cleaning passes.
    # This also detects a shared test image whose duplicate representative was
    # assigned to training/validation in either branch.
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        parent[find(left)] = find(right)

    for branch in branches:
        for row in branch['dedup'].itertuples(index=False):
            union(int(row.dropped_source_row_index), int(row.rep_source_row_index))
    used_roots = set()
    for branch in branches:
        used = pd.concat([branch['manifests']['train'], branch['manifests']['val']])
        if common & set(used.image_id):
            raise ValueError('Intersection contains training/validation images')
        used_roots.update(find(int(i)) for i in used.source_row_index)
    for branch in branches:
        shared = branch['manifests']['test']
        shared = shared[shared.image_id.isin(common)]
        if any(find(int(i)) in used_roots for i in shared.source_row_index):
            raise ValueError('Duplicate cluster from the test intersection appears in train/validation')

    left = a['predictions'][a['predictions'].image_id.isin(common)].sort_values('image_id').reset_index(drop=True)
    right = b['predictions'][b['predictions'].image_id.isin(common)].sort_values('image_id').reset_index(drop=True)
    if not np.array_equal(left.y_true.to_numpy(), right.y_true.to_numpy()):
        raise ValueError('Shared images have conflicting true labels; McNemar was not computed')
    for branch in branches:
        shared = branch['manifests']['test']
        for soft in shared.loc[shared.image_id.isin(common), 'soft_label']:
            votes = np.fromstring(str(soft), sep=' ')
            if len(votes) != len(label_names) or (votes == votes.max()).sum() != 1:
                raise ValueError('Intersection must have consistent untied reference labels')
    ids = left[['image_id', 'source_row_index', 'y_true']].rename(columns={'source_row_index': 'source_row_drop'})
    ids['source_row_keep'] = right.source_row_index.to_numpy()
    ids.to_csv(out / 'test_intersection_ids.csv', index=False)
    left.to_csv(out / 'predictions_drop_ties_intersection.csv', index=False)
    right.to_csv(out / 'predictions_keep_ties_intersection.csv', index=False)
    distribution = []
    for scope, frame in [('drop_ties_own_test', a['predictions']),
                         ('keep_ties_own_test', b['predictions']), ('intersection', left)]:
        for c, label in enumerate(label_names):
            count = int((frame.y_true == c).sum())
            distribution.append(dict(scope=scope, label_id=c, label=label, count=count,
                                     fraction=count / len(frame) if len(frame) else np.nan))
    pd.DataFrame(distribution).to_csv(out / 'intersection_class_distribution.csv', index=False)
    pd.DataFrame([dict(branch=x['name'], own_test_n=len(x['predictions']), intersection_n=len(common),
                      coverage=len(common) / len(x['predictions']) if len(x['predictions']) else np.nan,
                      protocol=protocol) for x in branches]).to_csv(out / 'intersection_coverage.csv', index=False)
    if not common:
        status = pd.DataFrame([dict(status='not_computed_empty_test_intersection', n=0, p_exact=np.nan)])
        for file in ('mcnemar_primary.csv', 'mcnemar_table_primary.csv',
                     'paired_predictions_primary.csv', 'comparison_vs_baseline_primary.csv'):
            status.to_csv(out / file, index=False)
        return status
    # baseline/old = drop_ties; candidate/new = keep_ties. No claim about all test images.
    return compare_paired_predictions(left, right, out, stem='primary')
