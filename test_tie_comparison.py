import json
import numpy as np
import pandas as pd
import pytest
import torch
from comparison_utils import compare_paired_predictions, export_predictions
from data_utils import load_fer2013_csv, extend_train_with_ties, load_split_from_manifest, EMOTION_COLS
from test_data_utils import _create_synthetic_fer_csvs
from eval_utils import SoftCrossEntropyLoss
from dataset import FER2013Dataset, mixup_soft_batch


def test_mcnemar_pairs_by_id_and_exact_p(tmp_path):
    a = pd.DataFrame({'image_id':list('abcdefgh'), 'y_true':[0]*8, 'y_pred':[1]*8})
    b = a.copy(); b['y_pred'] = 0
    result = compare_paired_predictions(a, b.iloc[::-1], tmp_path)
    assert result.iloc[0].new_only_correct == 8
    assert result.iloc[0].p_exact == pytest.approx(2/256)
    assert result.iloc[0].delta_accuracy == 1
    assert pd.isna(result.iloc[1].p_exact)
    same = compare_paired_predictions(a,a,tmp_path,'same')
    assert same.iloc[0].p_exact == 1
    with pytest.raises(ValueError): compare_paired_predictions(a,b.iloc[:-1],tmp_path)
    b.loc[0,'y_true']=2
    with pytest.raises(ValueError): compare_paired_predictions(a,b,tmp_path)


def test_frozen_extension_and_roundtrip(tmp_path):
    img, lbl = _create_synthetic_fer_csvs(tmp_path)
    images, labels = pd.read_csv(img), pd.read_csv(lbl)
    rng = np.random.default_rng(54)
    # New valid tie, exact duplicate of that tie, and a tie overlapping baseline.
    pixels = ' '.join(map(str,rng.integers(0,256,2304)))
    for pixel_string in [pixels,pixels,images.iloc[50].pixels]:
        src = len(labels)
        images.loc[len(images)] = {'Usage':'Training','pixels':pixel_string}
        labels.loc[src] = {'Usage':'Training','Image name':f'fer{src:07d}.png',
                          **{c:5 if c in ['neutral','contempt'] else 0 for c in EMOTION_COLS},'unknown':0,'NF':0}
    images.to_csv(img,index=False); labels.to_csv(lbl,index=False)
    baseline = tmp_path/'baseline'; output = tmp_path/'new'
    original = load_fer2013_csv(img,lbl,return_soft_labels=True,return_sample_weights=True,
                               split_manifest_dir=baseline,allow_positional_join=True)
    updated = extend_train_with_ties(img,lbl,baseline,output)
    assert len(updated[0][0]) == len(original[0][0])+1
    for split in ['val','test']:
        assert (baseline/f'{split}_ids.csv').read_bytes() == (output/f'{split}_ids.csv').read_bytes()
    for i in [1,2]:
        for j in range(4): np.testing.assert_array_equal(original[i][j], updated[i][j])
    np.testing.assert_allclose(updated[0][1][-1],[.5,0,0,0,0,0,0,.5])
    reloaded = load_split_from_manifest(img,lbl,output)
    for i in range(3):
        for j in range(4): np.testing.assert_array_equal(updated[i][j],reloaded[i][j])
    report = pd.read_csv(output/'tie_analysis.csv')
    assert (report.status=='added_train').sum()==1
    assert (report.status=='duplicate_exact').sum()==2
    manifest = pd.read_csv(output/'test_ids.csv')
    probs = np.eye(8)[updated[2][2]]
    exported = export_predictions(manifest,updated[2][2],probs,output,'test',EMOTION_COLS)
    assert exported.correct.all()


def test_cel_and_tied_rare_augmentation():
    logits = torch.tensor([[1.,0.,0.,0.,0.,0.,0.,-1.]],requires_grad=True)
    targets = torch.tensor([[.5,0,0,0,0,0,0,.5]])
    loss=SoftCrossEntropyLoss()(logits,targets)
    expected=-(targets*logits.log_softmax(1)).sum()
    torch.testing.assert_close(loss,expected)
    loss.backward(); assert torch.isfinite(logits.grad).all()
    class FailIfCalled:
        def __call__(self,x): raise AssertionError('Rare tie should be protected')
    ds=FER2013Dataset(np.zeros((1,48,48,1)),targets.numpy(),is_train=True)
    ds.spatial_transform=lambda x:x
    ds.erasing_transform=FailIfCalled()
    ds[0]
    images=torch.rand(2,1,48,48)
    labels=targets.repeat(2,1)
    mixed,soft=mixup_soft_batch(images,labels,alpha=.2,p=1)
    torch.testing.assert_close(mixed,images)
    torch.testing.assert_close(soft.sum(1),torch.ones(2))


def test_gradcam_explains_requested_prediction():
    from models import CustomCNN
    from eval_utils import generate_gradcam
    torch.set_num_threads(2)
    model = CustomCNN().eval()
    image = torch.rand(1,48,48)
    cam, target = generate_gradcam(model,image,target_class=7)
    assert target == 7
    assert cam.shape == (48,48)
    assert np.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1
