import random
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from torch.testing import assert_close
from torch.utils.data import Dataset

from train_rnn import KarpathyLSTM, NextByteDataset, batchify, read_corpus_bytes


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

    def assertBatchedDatasetsEqual(self, actual, expected):
        # Sadly a simple assertEqual doesn't work with lists containing Tensors, but hopefully this
        # is clear enough.

        self.assertEqual(len(actual), len(expected), "Number of batches mismatch")
        for i, (a, e) in enumerate(zip(actual, expected)):
            # Unpack flexibly: allow 2-tuple (ids only) or 4-tuple (ids + raw)
            self.assertIn(len(a), (2, 4), f"Unexpected tuple length in actual batch {i}")
            self.assertIn(len(e), (2, 4), f"Unexpected tuple length in expected batch {i}")

            ax, ay, *a_raw = a
            ex, ey, *e_raw = e

            # shapes/dtypes
            self.assertEqual(ax.shape, ex.shape, f"x shape mismatch in batch {i}")
            self.assertEqual(ay.shape, ey.shape, f"y shape mismatch in batch {i}")
            self.assertEqual(ax.dtype, ex.dtype, f"x dtype mismatch in batch {i}")
            self.assertEqual(ay.dtype, ey.dtype, f"y dtype mismatch in batch {i}")

            # values
            assert_close(ax, ex, msg=f"x values mismatch in batch {i}")
            assert_close(ay, ey, msg=f"y values mismatch in batch {i}")

            # raw fields present in both or neither
            self.assertEqual(len(a_raw), len(e_raw), f"raw field presence differs in batch {i}")
            if a_raw and e_raw:
                (axs_raw, ays_raw), (exs_raw, eys_raw) = a_raw, e_raw
                self.assertEqual(axs_raw, exs_raw, f"xs_raw mismatch in batch {i}")
                self.assertEqual(ays_raw, eys_raw, f"ys_raw mismatch in batch {i}")


    def test_sequences_are_contiguous_across_positions_in_batches(self):
        dataset = [
            (
                torch.tensor([ 1,  2,  3,  4], dtype=torch.long),
                torch.tensor([ 2,  3,  4,  5], dtype=torch.long),
                b"Sequence 1 xs",
                b"Sequence 1 ys",
            ),
            (
                torch.tensor([ 5,  6,  7,  8], dtype=torch.long),
                torch.tensor([ 6,  7,  8,  9], dtype=torch.long),
                b"Sequence 2 xs",
                b"Sequence 2 ys",
            ),
            (
                torch.tensor([ 9, 10, 11, 12], dtype=torch.long),
                torch.tensor([10, 11, 12, 13], dtype=torch.long),
                b"Sequence 3 xs",
                b"Sequence 3 ys",
            ),
            (
                torch.tensor([13, 14, 15, 16], dtype=torch.long),
                torch.tensor([14, 15, 16, 17], dtype=torch.long),
                b"Sequence 4 xs",
                b"Sequence 4 ys",
            ),
            (
                torch.tensor([17, 18, 19, 20], dtype=torch.long),
                torch.tensor([18, 19, 20, 21], dtype=torch.long),
                b"Sequence 5 xs",
                b"Sequence 5 ys",
            ),
            (
                torch.tensor([21, 22, 23, 24], dtype=torch.long),
                torch.tensor([22, 23, 24, 25], dtype=torch.long),
                b"Sequence 6 xs",
                b"Sequence 6 ys",
            ),
        ]

        batches = batchify(dataset, batch_size=3)

        # What we want is for all of the xs at position 0 in each
        # batch to be a contiguous sequence from the original xs,
        # and likewise for the ys, and so on for all other batch
        # positions.
        expected = [
            (
                torch.tensor(
                    [
                        [ 1,  2,  3,  4],
                        [ 9, 10, 11, 12],
                        [17, 18, 19, 20],
                    ],
                    dtype=torch.long
                ),
                torch.tensor(
                    [
                        [ 2,  3,  4,  5],
                        [10, 11, 12, 13],
                        [18, 19, 20, 21],
                    ],
                    dtype=torch.long
                ),
                [
                    b"Sequence 1 xs",
                    b"Sequence 3 xs",
                    b"Sequence 5 xs"
                ],
                [
                    b"Sequence 1 ys",
                    b"Sequence 3 ys",
                    b"Sequence 5 ys"
                ],
            ),
            (
                torch.tensor(
                    [
                        [ 5,  6,  7,  8],
                        [13, 14, 15, 16],
                        [21, 22, 23, 24],
                    ],
                    dtype=torch.long
                ),
                torch.tensor(
                    [
                        [ 6,  7,  8,  9],
                        [14, 15, 16, 17],
                        [22, 23, 24, 25],
                    ],
                    dtype=torch.long
                ),
                [
                    b"Sequence 2 xs",
                    b"Sequence 4 xs",
                    b"Sequence 6 xs"
                ],
                [
                    b"Sequence 2 ys",
                    b"Sequence 4 ys",
                    b"Sequence 6 ys"
                ],
            ),
        ]

        self.assertBatchedDatasetsEqual(batches, expected)


    def test_dataset_not_an_even_number_of_batches(self):
        dataset = [
            (torch.tensor([1], dtype=torch.long),
             torch.tensor([2], dtype=torch.long),
             b"xs1",
             b"ys1"),
            (torch.tensor([3], dtype=torch.long),
             torch.tensor([4], dtype=torch.long),
             b"xs2",
             b"ys2"),
            (torch.tensor([5], dtype=torch.long),
             torch.tensor([6], dtype=torch.long),
             b"xs3",
             b"ys3"),
        ]

        batches = batchify(dataset, batch_size=2)

        # With 3 items and batch_size=2, we only get 1 full batch
        expected = [
            (
                torch.tensor([[1], [3]], dtype=torch.long),
                torch.tensor([[2], [4]], dtype=torch.long),
                [b"xs1", b"xs2"],
                [b"ys1", b"ys2"],
            )
        ]

        self.assertBatchedDatasetsEqual(batches, expected)



class KarpathyLSTMTest(TestCase):

    def test_output_shapes(self):
        batch_size = 2
        vocab_size = 7
        sequence_length = 3

        def generate_random_input_batch():
            return torch.tensor(
                [
                    random.sample(range(vocab_size), sequence_length)
                    for _ in range(batch_size)
                ],
                dtype=torch.long
            )

        def assert_outputs_sane(logprobs, h_n, c_n):
            self.assertEqual(logprobs.shape, (batch_size, sequence_length, vocab_size))
            self.assertEqual(h_n.shape, (num_layers, batch_size, hidden_size))
            self.assertEqual(c_n.shape, (num_layers, batch_size, hidden_size))

            self.assertEqual(logprobs.dtype, torch.float)
            self.assertEqual(h_n.dtype, torch.float)
            self.assertEqual(c_n.dtype, torch.float)

            self.assertEqual(logprobs.device, torch.device(device))
            self.assertEqual(h_n.device, torch.device(device))
            self.assertEqual(c_n.device, torch.device(device))

            for batch_num in range(batch_size):
                for sequence_num in range(sequence_length):
                    self.assertAlmostEqual(torch.exp(logprobs[batch_num, sequence_num]).sum().item(), 1, delta=0.01)


        hidden_size = 7
        num_layers = 3
        dropout = 0.5
        device = "cpu"
        input_batch = generate_random_input_batch()

        model = KarpathyLSTM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.to(device)

        logprobs, (h_n, c_n) = model(input_batch)

        assert_outputs_sane(logprobs, h_n, c_n)

        new_batch = generate_random_input_batch()
        new_logprobs, (new_h_n, new_c_n) = model(new_batch, (h_n.detach(), c_n.detach()))

        assert_outputs_sane(new_logprobs, new_h_n, new_c_n)
































