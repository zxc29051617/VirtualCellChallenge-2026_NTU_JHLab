"""scDFM conditioned on the target's measured effect in other cell lines.

Identical to src/models/origin/model.model except for the perturbation input:
  * each gene token also receives the effect of the perturbation on that gene measured in other
    lines (log1p(CPM_pert/100) - log1p(CPM_ctrl/100)), encoded like an expression value and added
    to the control-cell channel;
  * the per-cell perturbation vector, which the original takes from the perturbed gene's learned
    token embedding, is a pooled summary of those per-gene effect tokens instead.
There is no per-perturbation identity left in the input, so the model cannot memorise genes: it has
to map "effect elsewhere + this line's control cells" to "effect here".
"""
import torch
import torch.nn as nn

from src.models.origin.layers import ContinuousValueEncoder
from src.models.origin.model import model as OriginModel


class EffectModel(OriginModel):
    def __init__(self, *args, dropout: float = 0.1, **kw):
        super().__init__(*args, dropout=dropout, **kw)
        d = self.t_embedder.mlp[2].weight.shape[0]
        self.effect_encoder = ContinuousValueEncoder(d, dropout)
        self.effect_pool = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, gene_id, cell_1, t, cell_2, effect, mode="predict_y"):
        if t.dim() == 0:
            t = t.repeat(cell_1.size(0))
        gene_emb = self.encoder(gene_id)
        eff_emb = self.effect_encoder(effect)
        value_emb_1 = self.value_encoder_1(cell_1) + gene_emb
        value_emb_2 = self.value_encoder_2(cell_2) + gene_emb + eff_emb
        x = self.fusion_layer(torch.cat([value_emb_1, value_emb_2], dim=-1))
        t_emb = self.t_embedder(t)
        perturbation_emb = self.effect_pool((gene_emb + eff_emb).mean(1))
        for i, block in enumerate(self.blocks):
            x = self.gene_adaLN[i](gene_emb, x)
            x = torch.cat([x, perturbation_emb[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
            x = self.adapter_layer[i](x)
            x = block(x, value_emb_2, t_emb)
        x = torch.cat([x, perturbation_emb[:, None, :].expand(-1, x.size(1), -1)], dim=-1)
        return self.final_layer(x)["pred"]
