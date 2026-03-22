"""Test that the Pro plugin registers correctly."""

from neural_memory_pro import __version__
from neural_memory_pro.plugin import NMProPlugin


def test_version():
    assert __version__ == "0.1.0"


def test_plugin_name():
    plugin = NMProPlugin()
    assert plugin.name == "neural-memory-pro"
    assert plugin.version == "0.1.0"


def test_retrieval_strategies():
    plugin = NMProPlugin()
    strategies = plugin.get_retrieval_strategies()
    assert "cone" in strategies


def test_compression_fn():
    plugin = NMProPlugin()
    fn = plugin.get_compression_fn()
    assert fn is not None


def test_consolidation_strategies():
    plugin = NMProPlugin()
    strategies = plugin.get_consolidation_strategies()
    assert "smart_merge" in strategies
