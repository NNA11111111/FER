from pathlib import Path
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
