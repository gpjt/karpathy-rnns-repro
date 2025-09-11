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
        seq_length = 4
        batch_size = 3
        num_batches = 4
        dataset_total_length = seq_length * batch_size * num_batches

        dataset_total_x_ids = torch.tensor(range(0, dataset_total_length), dtype=torch.long)
        dataset_total_y_ids = torch.tensor(range(1, dataset_total_length + 1), dtype=torch.long)

        test_x_ids = dataset_total_x_ids.split(seq_length)
        test_y_ids = dataset_total_y_ids.split(seq_length)

        dataset = [(xs, ys, f"Xs {ii}".encode(), f"Ys {ii}".encode()) for (ii, (xs, ys)) in enumerate(zip(test_x_ids, test_y_ids))]

        batches = batchify(dataset, batch_size)

        self.assertEqual(len(batches), num_batches)

        # What we want is for all of the xs at position 0 in each
        # batch to be a contiguous sequence from the original xs,
        # and likewise for the ys, and so on for all other batch
        # positions.
        batch_x_sequences = [[] for _ in range(batch_size)]
        batch_y_sequences = [[] for _ in range(batch_size)]
        for batch_xs, batch_ys in batches:
            self.assertEqual(batch_xs.dtype, torch.long)
            self.assertEqual(batch_xs.shape, (batch_size, seq_length))
            self.assertEqual(batch_ys.dtype, torch.long)
            self.assertEqual(batch_ys.shape, (batch_size, seq_length))
            for ii, item in enumerate(batch_xs):
                batch_x_sequences[ii].append(item)
            for ii, item in enumerate(batch_ys):
                batch_y_sequences[ii].append(item)

        for batch_position, batch_x_sequence in enumerate(batch_x_sequences):
            sequences_for_position = torch.stack(batch_x_sequence)
            start_in_original = batch_position * num_batches
            assert_close(
                sequences_for_position,
                torch.stack(test_x_ids[start_in_original:start_in_original + num_batches])
            )
        for batch_position, batch_y_sequence in enumerate(batch_y_sequences):
            sequences_for_position = torch.stack(batch_y_sequence)
            start_in_original = batch_position * num_batches
            assert_close(
                sequences_for_position,
                torch.stack(test_y_ids[start_in_original:start_in_original + num_batches])
            )


    def test_dataset_not_an_even_number_of_batches(self):
        self.fail("TODO")





































