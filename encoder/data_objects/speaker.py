"""
Contains classes and functions relavant to speakers
"""

from pathlib import Path
from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance

class Speaker:
    """
    Contains speaker info such as the name of the speaker, collection of utterances, and
    where to find those utterances
    """
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None

    def _load_utterances(self):
        with self.root.joinpath("_sources.txt").open("r") as sources_file:
            sources = [l.split(",") for l in sources_file]
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count: int, n_frames: int):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all
        utterances come up at least once every two cycles and in a random order every time.

        :param count: The number of partial utterances to sample from the set of utterances from
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance,
        frames are the frames of the partial utterances and range is the range of the partial
        utterance with regard to the complete utterance.
        """

        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        return [(u,) + u.random_partial(n_frames) for u in utterances]
