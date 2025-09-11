import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from torch.testing import assert_close
from torch.utils.data import Dataset

from train_rnn import NextByteDataset, batchify, read_corpus_bytes


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
        self.assertEqual(ds.id_to_byte[0], b"a"[0])
        self.assertEqual(ds.id_to_byte[1], b"d"[0])
        self.assertEqual(ds.id_to_byte[2], b"e"[0])


    def test_works_out_int_byte_to_id_mapping(self):
        ds = NextByteDataset(
            b"eeadaf",
            2
        )
        self.assertEqual(ds.byte_to_id[b"a"[0]], 0)
        self.assertEqual(ds.byte_to_id[b"d"[0]], 1)
        self.assertEqual(ds.byte_to_id[b"e"[0]], 2)


    def test_length_is_number_of_full_sequences(self):
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
        self.assertEqual(ys, b"da")
        assert_close(y_ids, torch.tensor([1, 0], dtype=torch.long))



class ReadCorpusBytesTest(TestCase):

    def test_barfs_if_not_input_txt_in_provided_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                read_corpus_bytes(tmp_dir)


    def test_reads_bytes_from_input_txt_in_provided_directory(self):
        some_bytes = b"sajdkfl\xff\x00\x80\x80fjdsalfsjkldjksfldjfds"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "input.txt").write_bytes(some_bytes)

            data = read_corpus_bytes(tmp_dir)

            self.assertEqual(data, some_bytes)



class BatchifyTest(TestCase):

    def test_sequences_are_contiguous_across_positions_in_batches(self):
        dataset = [
            (
                torch.tensor([ 1,  2,  3,  4], dtype=torch.long),
                torch.tensor([ 2,  3,  4,  5], dtype=torch.long),
                "Sequence 1 xs",
                "Sequence 1 ys",
            ),
            (
                torch.tensor([ 5,  6,  7,  8], dtype=torch.long),
                torch.tensor([ 6,  7,  8,  9], dtype=torch.long),
                "Sequence 2 xs",
                "Sequence 2 ys",
            ),
            (
                torch.tensor([ 9, 10, 11, 12], dtype=torch.long),
                torch.tensor([10, 11, 12, 13], dtype=torch.long),
                "Sequence 3 xs",
                "Sequence 3 ys",
            ),
            (
                torch.tensor([13, 14, 15, 16], dtype=torch.long),
                torch.tensor([14, 15, 16, 17], dtype=torch.long),
                "Sequence 4 xs",
                "Sequence 4 ys",
            ),
            (
                torch.tensor([17, 18, 19, 20], dtype=torch.long),
                torch.tensor([18, 19, 20, 21], dtype=torch.long),
                "Sequence 5 xs",
                "Sequence 5 ys",
            ),
            (
                torch.tensor([21, 22, 23, 24], dtype=torch.long),
                torch.tensor([22, 23, 24, 25], dtype=torch.long),
                "Sequence 6 xs",
                "Sequence 6 ys",
            ),
        ]

        batches = batchify(dataset, batch_size=3)

        # What we want is for all of the xs at position 0 in each
        # batch to be a contiguous sequence from the original xs,
        # and likewise for the ys, and so on for all other batch
        # positions.
        expected = [
            (
                torch.tensor([
                    [ 1,  2,  3,  4],
                    [ 9, 10, 11, 12],
                    [17, 18, 19, 20],
                ]),
                torch.tensor([
                    [ 2,  3,  4,  5],
                    [10, 11, 12, 13],
                    [18, 19, 20, 21],
                ]),
            ),
            (
                torch.tensor([
                    [ 5,  6,  7,  8],
                    [13, 14, 15, 16],
                    [21, 22, 23, 24],
                ]),
                torch.tensor([
                    [ 6,  7,  8,  9],
                    [14, 15, 16, 17],
                    [22, 23, 24, 25],
                ]),
            ),
        ]

        # Sadly a simple assertEqual doesn't work with lists containing Tensors, but hopefully this
        # is clear enough.
        self.assertEqual(len(batches), len(expected))
        for (actual_x, actual_y), (expected_x, expected_y) in zip(batches, expected):
            self.assertEqual(actual_x.shape, expected_x.shape)
            self.assertEqual(actual_y.shape, expected_y.shape)
            self.assertEqual(actual_x.dtype, torch.long)
            self.assertEqual(actual_y.dtype, torch.long)

            assert_close(actual_x, expected_x)
            assert_close(actual_y, expected_y)


    def test_x_and_y_raw_stuff(self):
        self.fail("TODO")


    def test_dataset_not_an_even_number_of_batches(self):
        self.fail("TODO")





































