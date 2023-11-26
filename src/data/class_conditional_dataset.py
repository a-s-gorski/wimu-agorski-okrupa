"""
Class Conditional Dataset Module

This module defines a PyTorch Dataset class, 'ClassConditionalDataset', representing a class-conditional dataset.
The dataset is designed to support class-conditional sampling and provide convenient access to information
about classes within the dataset.

Classes:
    ClassConditionalDataset(torch.utils.data.Dataset):
        A PyTorch Dataset representing a class-conditional dataset.

        Methods:
            __getitem__(index: int) -> Dict[Any, Any]:
                Grab an item from the dataset. The item returned must be a dictionary.

        Properties:
            classlist() -> List[str]:
                Returns a list of class labels available in the dataset.
                Enables users to easily access all the classes in the dataset.

            class_to_indices() -> Dict[str, List[int]]:
                Returns a dictionary where keys are class labels, and values are
                lists of indices in the dataset that belong to that class.
                Enables users to easily access examples belonging to specific classes.
"""
from typing import Any, Dict, List

import torch


class ClassConditionalDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset representing a class-conditional dataset.

    This dataset is designed to support class-conditional sampling and provide
    convenient access to information about classes within the dataset.

    Methods:
        __getitem__(index: int) -> Dict[Any, Any]:
            Grab an item from the dataset. The item returned must be a dictionary.

        Properties:
            classlist() -> List[str]:
                Returns a list of class labels available in the dataset.
                Enables users to easily access all the classes in the dataset.

            class_to_indices() -> Dict[str, List[int]]:
                Returns a dictionary where keys are class labels, and values are
                lists of indices in the dataset that belong to that class.
                Enables users to easily access examples belonging to specific classes.
    """

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """
        Grab an item from the dataset. The item returned must be a dictionary.
        """
        raise NotImplementedError

    @property
    def classlist(self) -> List[str]:
        """
        The classlist property returns a list of class labels available in the dataset.
        This property enables users of the dataset to easily access a list of all the classes in the dataset.

        Returns:
            List[str]: A list of class labels available in the dataset.
        """
        raise NotImplementedError

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        """
        Returns a dictionary where the keys are class labels and the values are
        lists of indices in the dataset that belong to that class.
        This property enables users of the dataset to easily access
        examples that belong to specific classes.

        Implement me!

        Returns:
            Dict[str, List[int]]: A dictionary mapping class labels to lists of dataset indices.
        """
        raise NotImplementedError
