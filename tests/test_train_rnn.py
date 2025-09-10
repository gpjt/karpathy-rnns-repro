from unittest import TestCase

import torch
from torch.testing import assert_close
from torch.utils.data import Dataset

from train_rnn import NextByteDataset


class NextByteDatasetTest(TestCase):

    def test_is_dataset(self):
        ds = NextByteDataset(b"ab", 1)
        self.assertTrue(isinstance(ds, Dataset))


    def test_barfs_on_empty_sequence_length(self):
        with self.assertRaises(AssertionError):
            NextByteDataset(b"", 0)
        with self.assertRaises(AssertionError):
            NextByteDataset(b"", -1)


    def test_stashes_seq_length_on_self(self):
        ds = NextByteDataset(b"1" * 124, 123)
        self.assertEqual(ds.seq_length, 123)


    def test_barfs_if_no_sequences(self):
        # Note that we need not just the X sequences, which start at position 0, but also the Y
        # sequences.  So we need one more than 1 * seq_length
        with self.assertRaises(AssertionError):
            NextByteDataset(b"abc", 3)


    def test_puts_all_sequences_plus_one_byte_into_data(self):
        ds = NextByteDataset(
            b"abcdef",
            2
        )
        self.assertEqual(
            ds.data,
            b"abcde",
        )


    def test_works_out_vocab_as_sorted_list_of_ints_based_on_trimmed_data(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        self.assertEqual(
            ds.vocab,
            list(b"ade"),
        )


    def test_works_out_id_to_byte_mapping_with_bytes_as_ints(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        self.assertEqual(ds.id_to_byte[0], ord(b"a"))
        self.assertEqual(ds.id_to_byte[1], ord(b"d"))
        self.assertEqual(ds.id_to_byte[2], ord(b"e"))


    def test_works_out_int_byte_to_id_mapping(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        self.assertEqual(ds.byte_to_id[ord(b"a")], 0)
        self.assertEqual(ds.byte_to_id[ord(b"d")], 1)
        self.assertEqual(ds.byte_to_id[ord(b"e")], 2)


    def test_creates_tensor_with_ids(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        assert_close(
            ds.data_as_ids,
            torch.tensor([2, 2, 0, 1, 0], dtype=torch.long)
        )


    def test_length_is_number_of_sequences(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        self.assertEqual(len(ds), 2)


    def test_can_get_sequences(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        seq_1 = ds[0]
        x_ids, y_ids, xs, ys = seq_1
        self.assertEqual(xs, b"ee")
        assert_close(x_ids, torch.tensor([2, 2], dtype=torch.long))
        self.assertEqual(ys, b"ea")
        assert_close(y_ids, torch.tensor([2, 0], dtype=torch.long))

        seq_2 = ds[1]
        x_ids, y_ids, xs, ys = seq_2
        self.assertEqual(xs, b"ad")
        assert_close(x_ids, torch.tensor([0, 1], dtype=torch.long))
        self.assertEqual(ys, b"da]")
        assert_close(y_ids, torch.tensor([1, 0], dtype=torch.long))


