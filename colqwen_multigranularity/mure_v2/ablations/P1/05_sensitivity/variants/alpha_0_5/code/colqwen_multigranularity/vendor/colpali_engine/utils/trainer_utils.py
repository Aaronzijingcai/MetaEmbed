from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    ProgressCallback,
)


def update_logs_with_losses(logs, state):
    losses = getattr(state, "loss_stats", None)
    if losses is not None:
        logs.update(losses)
    # else:
    #     print(f"You used a `WithLosses` Callback but no sub losses are reported!")
    return logs


class ProgressCallbackWithLosses(ProgressCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "loss_stats"):
            state.loss_stats = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            logs = update_logs_with_losses(logs, state)
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))


class PrinterCallbackWithLosses(PrinterCallback):
    # hack some necessary callbacks is enough
    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if not hasattr(state, "loss_stats"):
            state.loss_stats = dict()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = update_logs_with_losses(logs, state)
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


REPLACE_MAPPING = {
    # WandbCallback: WandbCallbackWithLosses,
    ProgressCallback: ProgressCallbackWithLosses,
    PrinterCallback: PrinterCallbackWithLosses,
}


def hack_callbacks_and_replace(callback_handler: CallbackHandler):
    for i, callback in enumerate(callback_handler.callbacks):
        for ins in REPLACE_MAPPING.keys():
            if isinstance(callback, ins):
                callback_handler.callbacks[i] = REPLACE_MAPPING[ins]()
                break
    return callback_handler
