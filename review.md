# Review and implementation notes

## Current pipeline decisions

- The default stratified split is 60/30/10 using the majority hard label. `split_mode="original_usage"` preserves Training/PublicTest/PrivateTest roles after global deduplication.
- A tied-label evaluation array can be returned separately from the official PrivateTest partition. It is intended for probability metrics against FER+ soft vote distributions.
- Preprocessing defaults to 112×112. FER2013 pixels are still 48×48; interpolation changes model input dimensions but cannot restore missing detail.
- Target-class CAM extraction, flipped-CAM consistency, and an EAC augmentation mode (horizontal flip plus dynamic erasing for every class) are exposed. Notebook training code still needs to compute only the original-view classification loss and add the consistency term.
- ECE and scalar temperature fitting helpers are implemented. Temperature must be fitted on validation or out-of-fold logits, never the final test set.
- A face-pretrained ResNet18 adapter is available but requires a local compatible checkpoint. No weights or measured accuracy claims are included.

## Remaining experiment requirements

- Run controlled ablations: soft labels; + train-time logit adjustment; + one scalar post-hoc adjustment; + sampler; + class weights. Avoid multiplying sample reliability weights in both sampler and loss.
- Compare EAC on/off and random-erasing on/off, since erasing can be part of EAC and should not be treated as an independent guarantee of noise robustness.
- Fit calibration with 5-fold out-of-fold validation if the held-out validation set is too small; report ECE, soft-label ECE, NLL, Brier score, accuracy, macro-F1, and per-class precision/recall.
- Add a results table for Barsoum/FER+, EAC, and SCN. Label published and reproduced results separately and include each dataset, input resolution, split, and metric protocol.
- The project notebook has not been edited. Update it to use `split_mode="original_usage"`, retrieve the optional tie evaluation array, apply scalar `tau` only once, fit temperature/ECE, pass the local face checkpoint when using `FacePretrainedResNet18`, and serialize `temperature` plus `model_type` in the bundle.
- The source commit history was rewritten to remove `.pth` files. No force-push was performed; the origin remote has been restored. The host's private Codex checkpoint refs may still retain old snapshot objects outside ordinary commit history.
