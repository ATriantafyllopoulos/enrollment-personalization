import torch

from transformers import (
    Wav2Vec2Model
)

def create_model(cfg, output_dim):
    if cfg.encoder.name == "w2v2":
        encoder = W2V2Backbone(
            model_name=cfg.encoder.model_name
        )
    else:
        raise NotImplementedError(cfg.encoder.name)
    
    if cfg.frontend.name == "ffnn":
        frontend = FFNN(
            input_size=encoder.hidden_size,
            hidden_size=cfg.frontend.hidden_size,
            output_dim=output_dim,
            num_layers=cfg.frontend.num_layers,
            dropout=cfg.frontend.dropout,
        )
    else:
        raise NotImplementedError(cfg.frontend.name)

    if cfg.personalization.fusion.name == "attention":
        fusion = AttentionFusion(
            hidden_size=encoder.hidden_size,
            num_heads=cfg.personalization.fusion.num_heads
        )
    else:
        raise NotImplementedError(cfg.personalization.fusion.name)

    model = Model(
        encoder=encoder,
        frontend=frontend,
        fusion=fusion,
        enrollment=cfg.personalization.enrollment
    )
    return model

class AttentionFusion(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads
    ):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            batch_first=True,
            num_heads=num_heads
        )
    def forward(self, embeddings, adaptation):
        pass

class FFNN(torch.nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 2,
        sigmoid: bool = False,
        softmax: bool = False,
        dropout: float = 0.5
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.dropout = dropout

        layers = []
        layer_input = input_size
        for i in range(num_layers - 1):
            layers += [
                (f"Linear{i}", torch.nn.Linear(layer_input, hidden_size)),
                (f"ReLU{i}", torch.nn.ReLU()),
                (f"Dropout{i}", torch.nn.Dropout(dropout)),
            ]
            layer_input = hidden_size
        layers.append((
            f"Linear{num_layers - 1}",
            torch.nn.Linear(layer_input, output_dim)
        ))
        if self.sigmoid:
            layers.append(("Sigmoid", torch.nn.Sigmoid()))
        if self.softmax:
            layers.append(("Softmax", torch.nn.Softmax(dim=1)))

        for name, layer in layers:
            self.add_module(name, layer)

class W2V2Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name,
        freeze_extractor: bool = True,
        time_pooling: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.time_pooling = time_pooling
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.hidden_size = self.model.config.hidden_size
        if self.freeze_extractor:
            self.model.freeze_feature_encoder()

    def forward(self, x):
        x = self.model(x)["last_hidden_state"]
        if self.time_pooling:
            x = x.mean(1)
        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        encoder,
        frontend,
        fusion,
        enrollment
    ):
        super().__init__()
        self.encoder = encoder
        self.frontend = frontend
        self.fusion = fusion
        self.enrollment = enrollment

    def forward(self, data):
        results = {
            "instance": {},
            "enrollment": {}
        }
        embeddings = self.encoder(data["instance"]["input"])
        results["instance"]["output"] = self.frontend(embeddings)

        if self.enrollment is not None:
            pass
        
        return results
    
