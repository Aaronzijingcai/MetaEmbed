import importlib

if importlib.util.find_spec("transformers") is not None:
    from colpali_engine.models import ColPaliProcessor
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer

    class AllPurposeWrapper:
        def __new__(cls, class_to_instantiate, *args, **kwargs):
            return class_to_instantiate.from_pretrained(*args, **kwargs)

    class AutoProcessorWrapper:
        def __new__(cls, *args, **kwargs):
            return AutoProcessor.from_pretrained(*args, **kwargs)

    class AutoTokenizerWrapper(PreTrainedTokenizer):
        def __new__(cls, *args, **kwargs):
            return AutoTokenizer.from_pretrained(*args, **kwargs)

    class ColPaliProcessorWrapper:
        def __new__(cls, *args, **kwargs):
            return ColPaliProcessor.from_pretrained(*args, **kwargs)

else:
    raise ModuleNotFoundError("Transformers must be loaded")
