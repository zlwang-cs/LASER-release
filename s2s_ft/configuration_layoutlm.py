from transformers import BertConfig


LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'layoutlm-base-uncased': 'https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/config.json',
    'layoutlm-large-uncased': 'https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/config.json'
}


class LayoutlmConfig(BertConfig):
    pretrained_config_archive_map = LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
