"""CPU regression checks for the small script-level fixes."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import eval as eval_script
import train as train_script
from datasets.sonicom_dataset import SonicomHLogDataset
from datasets.sparse_sets import get_sparse_set


def data_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroModel(torch.nn.Module):
    def forward(self, values):
        return torch.zeros_like(values)


class SmallFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_evaluation_weights_subjects_not_batches(self):
        levels = np.array([1, 1, 3, 3, 9], dtype=np.float32)
        values = np.broadcast_to(levels[:, None, None, None], (5, 2, 2, 4)).copy()
        dataset = SonicomHLogDataset(values, np.array([0, 1]), np.arange(5))
        for batch_size in (1, 2, 5):
            lsd, ild = eval_script.evaluate(
                ZeroModel(), DataLoader(dataset, batch_size=batch_size), [0, 1], "cpu"
            )
            self.assertAlmostEqual(lsd, 3.4, places=6)
            self.assertAlmostEqual(ild, 0.0, places=6)

    def test_evaluation_rejects_empty_loader(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            eval_script.evaluate(ZeroModel(), [], [0], "cpu")

    def test_sparse_layout_copies_preserve_order(self):
        for count in (3, 5, 19, 100):
            expected = get_sparse_set(count)
            changed = get_sparse_set(count)
            changed[0] = -1
            np.testing.assert_array_equal(get_sparse_set(count), expected)
            self.assertEqual(len(np.unique(expected)), count)

    def test_float32_storage_is_shared_and_samples_unchanged(self):
        values = np.arange(3 * 4 * 2 * 8, dtype=np.float32).reshape(3, 4, 2, 8)
        dataset = SonicomHLogDataset(values, np.array([2, 0]), np.array([1]))
        self.assertTrue(np.shares_memory(dataset.H, values))
        sparse, dense = dataset[0]
        np.testing.assert_array_equal(sparse.numpy(), values[1, [2, 0]])
        np.testing.assert_array_equal(dense.numpy(), values[1])

    def test_data_imports_perform_no_preparation(self):
        # Import dependencies before checking only the scripts' side effects.
        import sofar
        with patch("requests.get") as network, patch.object(sofar, "read_sofa") as read:
            with patch("numpy.savez") as save, patch("pathlib.Path.mkdir") as mkdir:
                data_script("download_sonicom")
                data_script("preprocess")
        network.assert_not_called()
        read.assert_not_called()
        save.assert_not_called()
        mkdir.assert_not_called()

    def test_preprocess_rejects_wrong_length_and_sampling_rate(self):
        module = data_script("preprocess")
        paths = [Path(f"P{i:04d}.sofa") for i in range(1, 201)]
        invalid = [
            SimpleNamespace(Data_IR=np.zeros((793, 2, 128)), Data_SamplingRate=48000),
            SimpleNamespace(Data_IR=np.zeros((793, 2, 256)), Data_SamplingRate=44100),
            SimpleNamespace(Data_IR=np.zeros((792, 2, 256)), Data_SamplingRate=48000),
        ]
        for sofa in invalid:
            with patch.object(Path, "glob", return_value=paths):
                with patch.object(module.sofar, "read_sofa", return_value=sofa):
                    with patch.object(module, "tqdm", side_effect=lambda x: x):
                        with self.assertRaises(ValueError):
                            module.preprocess(Path("."), Path("unused.npz"))

    def test_preprocess_preserves_fft_and_db_values(self):
        module = data_script("preprocess")
        paths = [Path(f"P{i:04d}.sofa") for i in range(1, 201)]
        hrir = np.zeros((793, 2, 256))
        hrir[:, 0, 0], hrir[:, 1, 0] = 1.0, 0.5
        sofa = SimpleNamespace(Data_IR=hrir, Data_SamplingRate=48000)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "glob", return_value=paths):
                with patch.object(module.sofar, "read_sofa", return_value=sofa):
                    with patch.object(module, "tqdm", side_effect=lambda x: x):
                        with patch.object(module.np, "savez") as save:
                            with contextlib.redirect_stdout(io.StringIO()):
                                result = module.preprocess(Path("."), Path(directory) / "out.npz")
        self.assertEqual(result.shape, (200, 793, 2, 106))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[:, :, 0], 0, atol=1e-6)
        np.testing.assert_allclose(result[:, :, 1], 20 * np.log10(0.5), atol=1e-6)
        self.assertIs(save.call_args.kwargs["H_log"], result)

    def test_failed_download_removes_partial_output(self):
        module = data_script("download_sonicom")
        response = SimpleNamespace(headers={})
        response.raise_for_status = lambda: None

        def interrupted(**kwargs):
            yield b"partial"
            raise requests.ConnectionError("interrupted")
        response.iter_content = interrupted
        from unittest.mock import MagicMock
        context = MagicMock()
        context.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "range", return_value=[1], create=True):
                with patch.object(module.requests, "get", return_value=context):
                    with patch.object(module, "tqdm", side_effect=lambda x, **kw: x):
                        with contextlib.redirect_stdout(io.StringIO()):
                            failed = module.download_sonicom(Path(directory))
            self.assertEqual(failed, ["P0001"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_eval_loads_tensor_weights_safely_and_strictly(self):
        config = eval_script.load_cfg(ROOT / "configs/default.yaml")
        config["device"] = "cpu"
        config["data"].update(n_subjects=4, train_subjects=2, n_freq=8, n_measurements=3)
        config["train"]["batch_size"] = 2
        config["model"].update(
            d_model=8, d_ff=16, n_heads=2, n_layers=1,
            conv_kernel=3, head_hidden=16, dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config["data"]["npz_path"] = str(directory / "data.npz")
            np.savez(config["data"]["npz_path"],
                     H_log=np.zeros((4, 793, 2, 8), dtype=np.float32))
            state = eval_script.build_model(config, 3).state_dict()
            checkpoint = directory / "best_model.pt"
            torch.save(state, checkpoint)
            original_load = torch.load
            with patch.object(eval_script, "load_cfg", return_value=config):
                with patch.object(eval_script, "seed_everything"):
                    with patch.object(torch, "load", wraps=original_load) as load:
                        with contextlib.redirect_stdout(io.StringIO()):
                            eval_script.main("unused.yaml", str(checkpoint))
                    self.assertIs(load.call_args.kwargs["weights_only"], True)
                    state.pop(next(iter(state)))
                    torch.save(state, checkpoint)
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, "Missing key"):
                            eval_script.main("unused.yaml", str(checkpoint))

    def test_training_monitor_is_subject_weighted(self):
        class TrainableZero(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.zeros(()))

            def forward(self, values):
                return values.repeat(1, 2, 1, 1) * self.scale

        config = train_script.load_cfg(ROOT / "configs/default.yaml")
        config["device"] = "cpu"
        config["data"].update(n_subjects=6, train_subjects=5, n_dirs=2, n_freq=2)
        config["train"].update(batch_size=2, epochs=1, eval_every=1)
        levels = np.array([1, 1, 3, 3, 9, 2], dtype=np.float32)
        values = np.broadcast_to(levels[:, None, None, None], (6, 2, 2, 2)).copy()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config["data"]["npz_path"] = str(directory / "data.npz")
            config["train"]["save_dir"] = str(directory / "runs")
            np.savez(config["data"]["npz_path"], H_log=values)
            output = io.StringIO()
            with patch.object(train_script, "load_cfg", return_value=config):
                with patch.object(train_script, "get_sparse_set", return_value=np.array([0])):
                    with patch.object(train_script, "HRTFMagUpsampler", TrainableZero):
                        with patch.object(train_script, "hrtf_shape_loss",
                                          side_effect=lambda pred, target, **kw:
                                          target.mean() + pred.sum() * 0):
                            with contextlib.redirect_stdout(output):
                                train_script.main("unused.yaml")
            best = next(line for line in output.getvalue().splitlines()
                        if line.startswith("Best train loss:"))
            self.assertAlmostEqual(float(best.split(":", 1)[1]), 3.4, places=6)
            self.assertEqual(len(list(directory.rglob("best_model.pt"))), 1)


if __name__ == "__main__":
    unittest.main()
