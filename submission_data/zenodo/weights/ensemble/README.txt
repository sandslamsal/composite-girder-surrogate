Deep-ensemble weights
=====================

This folder is intended to hold five independently trained residual-MLP
members:

    ensemble_member_1.pt
    ensemble_member_2.pt
    ensemble_member_3.pt
    ensemble_member_4.pt
    ensemble_member_5.pt

At the time of this Zenodo upload the ensemble had NOT been
pre-trained. The paper's uncertainty band (Fig. 9) uses MC-Dropout
inference on the headline model (`weights/headline_model.pt` in the
code archive). The deep-ensemble training script
`scripts/train_ensemble.py` is included in the code archive for
follow-up work and for users who require deployment-grade uncertainty
estimates.
