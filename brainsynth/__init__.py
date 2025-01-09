from pathlib import Path

root_dir = Path(__file__).parent
resources_dir = root_dir / "resources"

from brainsynth import config
from brainsynth import dataset
from brainsynth.synthesizer import Synthesizer
