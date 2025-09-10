from unittest import TestCase

import torch
from torch.testing import assert_close
from torch.utils.data import Dataset

from train_rnn import NextByteDataset


class NextByteDatasetTest(TestCase):

    def test_is_dataset(self):
        ds = NextByteDataset([1, 2], 1)
        self.assertTrue(isinstance(ds, Dataset))


    def test_barfs_on_empty_sequence_length(self):
        with self.assertRaises(AssertionError):
            NextByteDataset([], 0)
        with self.assertRaises(AssertionError):
            NextByteDataset([], -1)


    def test_stashes_seq_length_on_self(self):
        ds = NextByteDataset([1] * 124, 123)
        self.assertEqual(ds.seq_length, 123)


    def test_barfs_if_no_sequences(self):
        # Note that we need not just the X sequences, which start at position 0, but also the Y
        # sequences.  So we need one more than 1 * seq_length
        with self.assertRaises(AssertionError):
            NextByteDataset([1, 2, 3], 3)


    def test_puts_all_sequences_plus_one_byte_into_data(self):
        ds = NextByteDataset(
            [1, 2, 3, 4, 5, 6],
            2
        )
        self.assertEqual(
            ds.data,
            [1, 2, 3, 4, 5]
        )


    def test_works_out_vocab_as_sorted_list_based_on_trimmed_data(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        self.assertEqual(
            ds.vocab,
            [1, 4, 5]
        )


    def test_works_out_id_to_byte_mapping(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        self.assertEqual(ds.id_to_byte[0], 1)
        self.assertEqual(ds.id_to_byte[1], 4)
        self.assertEqual(ds.id_to_byte[2], 5)


    def test_works_out_byte_to_id_mapping(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        self.assertEqual(ds.byte_to_id[1], 0)
        self.assertEqual(ds.byte_to_id[4], 1)
        self.assertEqual(ds.byte_to_id[5], 2)


    def test_creates_tensor_with_ids(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        assert_close(
            ds.data_as_ids,
            torch.tensor([2, 2, 0, 1, 0], dtype=torch.long)
        )


    def test_length(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        self.assertEqual(len(ds), 2)


    def test_can_get_sequences(self):
        ds = NextByteDataset(
            [5, 5, 1, 4, 1, 6],
            2
        )
        seq_1 = ds[0]
        x_ids, y_ids, xs, ys = seq_1
        self.assertEqual(xs, [5, 5])
        assert_close(x_ids, torch.tensor([2, 2], dtype=torch.long))
        self.assertEqual(ys, [5, 1])
        assert_close(y_ids, torch.tensor([2, 0], dtype=torch.long))

        seq_2 = ds[1]
        x_ids, y_ids, xs, ys = seq_2
        self.assertEqual(xs, [1, 4])
        assert_close(x_ids, torch.tensor([0, 1], dtype=torch.long))
        self.assertEqual(ys, [4, 1])
        assert_close(y_ids, torch.tensor([1, 0], dtype=torch.long))


