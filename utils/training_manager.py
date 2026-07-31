import threading

# Global training state — one job at a time
_state: dict = {
    'running': False,
    'progress': 0,
    'status': '',
    'error': None,
    'result': None,
}
_lock = threading.Lock()


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kwargs):
    with _lock:
        _state.update(**kwargs)


def _progress_cb(epoch: int, total: int, logs: dict):
    pct = int((epoch / total) * 100)
    acc = logs.get('accuracy', 0)
    val_acc = logs.get('val_accuracy', 0)
    _set(progress=pct,
         status=f'Epoch {epoch}/{total} — acc: {acc:.3f} — val_acc: {val_acc:.3f}')


def start_ann_training(df, target_col: str, epochs: int, on_done=None) -> bool:
    """Start ANN training in a background thread. Returns False if already running."""
    with _lock:
        if _state['running']:
            return False
        _state.update(running=True, progress=0,
                      status='Preparing data…', error=None, result=None)

    def _run():
        try:
            from utils.ann_utils import train_ann_with_progress
            result = train_ann_with_progress(df, target_col, epochs, _progress_cb)
            _set(running=False, progress=100, status='Training complete!', result=result)
            if on_done:
                on_done(result)
        except Exception as e:
            _set(running=False, progress=0, status='Training failed.', error=str(e))

    threading.Thread(target=_run, daemon=True).start()
    return True
