"""
Episode Dataset Class

A dataset for sampling few-shot learning tasks from a class-conditional dataset.

Args:
    dataset (ClassConditionalDataset): The dataset to sample episodes from.
    n_way (int): The number of classes to sample per episode. Default: 5.
    n_support (int): The number of samples per class to use as support. Default: 5.
    n_query (int): The number of samples per class to use as query. Default: 20.
    n_episodes (int): The number of episodes to generate. Default: 100.

Methods:
    __getitem__(index: int) -> Tuple[Dict, Dict]:
        Sample an episode from the class-conditional dataset.
        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

    __len__() -> int:
        Return the number of episodes in the dataset.

    print_episode(support: Dict[str, Any], query: Dict[str, Any]) -> None:
        Print a summary of the support and query sets for an episode.
"""
import random
from typing import Dict, Tuple

import music_fsl.util as util
import torch
from music_fsl.data import ClassConditionalDataset


class EpisodeDataset(torch.utils.data.Dataset):
    """
        A dataset for sampling few-shot learning tasks from a class-conditional dataset.

    Args:
        dataset (ClassConditionalDataset): The dataset to sample episodes from.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 20.
        n_episodes (int): The number of episodes to generate.
            Default: 100.
    """

    def __init__(self,
                 dataset: ClassConditionalDataset,
                 n_way: int = 5,
                 n_support: int = 5,
                 n_query: int = 20,
                 n_episodes: int = 100,
                 ):
        self.dataset = dataset

        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """Sample an episode from the class-conditional dataset.

        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set and the query set for an episode.
        """
        # seed the random number generator so we can reproduce this episode
        rng = random.Random(index)

        # sample the list of classes for this episode
        episode_classlist = rng.sample(self.dataset.classlist, self.n_way)

        # sample the support and query sets for this episode
        support, query = [], []
        for c in episode_classlist:
            # grab the dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]
            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["target"] = torch.tensor(episode_classlist.index(c))

            # split the support and query sets
            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        # collate the support and query sets
        support = util.collate_list_of_dicts(support)
        query = util.collate_list_of_dicts(query)

        support["classlist"] = episode_classlist
        query["classlist"] = episode_classlist

        return support, query

    def __len__(self):
        return self.n_episodes

    def print_episode(self, support, query) -> None:
        """Print a summary of the support and query sets for an episode.

        Args:
            support (Dict[str, Any]): The support set for an episode.
            query (Dict[str, Any]): The query set for an episode.
        """
        print("Support Set:")
        print(f"  Classlist: {support['classlist']}")
        print(f"  Audio Shape: {support['audio'].shape}")
        print(f"  Target Shape: {support['target'].shape}")
        print()
        print("Query Set:")
        print(f"  Classlist: {query['classlist']}")
        print(f"  Audio Shape: {query['audio'].shape}")
        print(f"  Target Shape: {query['target'].shape}")
