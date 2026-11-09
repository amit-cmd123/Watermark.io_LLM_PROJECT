"""An installed model must load without contacting the network."""
import unittest
from unittest.mock import Mock, call

from model_loading import load_pretrained


class ModelLoadingTests(unittest.TestCase):
    def test_cached_model_never_requests_network(self):
        loader = Mock()
        model = load_pretrained(loader, 'example/model', 'commit', torch_dtype='dtype')
        self.assertIs(model, loader.from_pretrained.return_value)
        loader.from_pretrained.assert_called_once_with(
            'example/model', revision='commit', local_files_only=True, torch_dtype='dtype'
        )

    def test_missing_files_still_allow_download(self):
        loader = Mock()
        loader.from_pretrained.side_effect = [OSError('missing weights'), 'downloaded']
        self.assertEqual(load_pretrained(loader, 'example/model'), 'downloaded')
        self.assertEqual(loader.from_pretrained.call_args_list, [
            call('example/model', revision='main', local_files_only=True),
            call('example/model', revision='main'),
        ])

    def test_other_errors_are_not_retried_as_downloads(self):
        loader = Mock()
        loader.from_pretrained.side_effect = ValueError('unsupported architecture')
        with self.assertRaises(ValueError):
            load_pretrained(loader, 'example/model')
        self.assertEqual(loader.from_pretrained.call_count, 1)


if __name__ == '__main__':
    unittest.main()
