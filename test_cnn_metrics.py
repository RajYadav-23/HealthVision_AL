from utils.cnn_utils import load_cnn_metrics


def test_cnn_metrics_expose_summary_stats():
    metrics = load_cnn_metrics()
    assert metrics is not None
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
