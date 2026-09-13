"""Select one immutable XML variant for the lifetime of a CLI process."""
from pathlib import Path
from infantry_rl import mjlab_task as base


def select_model(variant='original'):
    filenames={'original':'training_scene.xml','fix':'training_scene_fix.xml'}
    if variant not in filenames: raise ValueError(f'Unknown model variant: {variant}')
    path=Path(base.__file__).parent/'robot/xmls'/filenames[variant]
    if not path.is_file(): raise FileNotFoundError(path)
    # Existing task functions/spec callbacks resolve base.MODEL when called.
    # This CLI-level selection must precede all config/environment construction.
    base.MODEL=path
    return path
