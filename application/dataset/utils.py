from sklearn.model_selection import train_test_split

from application.preprocessing.operations.chunking import chunk_document
from domain.cleaned_documents import CleanedDocument
from domain.dataset import (
  InstructDataset,
  InstructDatasetSample,
  InstructTrainTestSplit
)
from domain.types import DataCategory


def create_instruct_train_test_split(
  data: dict[DataCategory, InstructDataset], test_size=0.2, random_state=42
) -> InstructTrainTestSplit:
  train_data = {}
  test_data = {}

  for category, dataset in data.items():
    samples = dataset.samples
    samples_dicts = [sample.model_dump() for sample in samples]

    if len(samples_dicts) > 0:
      train_samples_dicts, test_samples_dicts = train_test_split(
        samples_dicts, test_size=test_size, random_state=random_state
      )
      train_samples = [InstructDatasetSample(**sample_dict) for sample_dict in train_samples_dicts]
      test_samples = [InstructDatasetSample(**sample_dict) for sample_dict in test_samples_dicts]
    else:
      train_samples = []
      test_samples = []

    train_dataset = InstructDataset(category=category, samples=train_samples)
    test_dataset = InstructDataset(category=category, samples=test_samples)

    train_data[category] = train_dataset
    test_data[category] = test_dataset

  return InstructTrainTestSplit(train=train_data, test=test_data, test_split_size=test_size)


def extract_substrings(
  documents: list[CleanedDocument], min_length: int = 1000, max_length: int = 2000
) -> list[CleanedDocument]:
  extracts = []
  for document in documents:
    document_extracts = chunk_document(document.content, min_length, max_length)
    for extract in document_extracts:
      subdocument = document.model_copy()
      subdocument.content = extract

      extracts.append(subdocument)

  return extracts