"""
Checkpoint upgrades of the FlashMD weak symplectic architecture.

The architecture starts at model version 1 and trainer version 1, on top of the
``MetatrainModel`` wrapper of metatrain (the scaler and the additive models live in the
wrapper, not in the model). There is nothing to upgrade yet: add the functions
``model_update_v{n}_v{n+1}`` / ``trainer_update_v{n}_v{n+1}`` here when the versions are
bumped.

Checkpoints of the older ``experimental.flashmd_weak_symplectic`` working copy (model
version 5, scaler and additive models inside the model) cannot be loaded by this
architecture.
"""
