"""Region names shared by the harness and its report.

A region is a tensor compiled vLLM stores inside a Grug decoder layer that the trainer also holds.
``REGIONS`` lists them in forward order. The role maps name the stored values of the archived
subgraphs: the piece holding the layer's pre-attention half (``next.`` roles) and the piece holding its
post-attention half (``prev.`` roles, plus the following layer's input norm).
"""

# Region names in forward order. Each is a tensor compiled vLLM stores and the trainer also holds.
REGIONS = (
    "input",
    "attn_rms",
    "attn_gate_down",
    "attn_gate_act",
    "attn_gate_up",
    "attention_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "query",
    "key",
    "core_attention",
    "attn_gate",
    "xsa_gate",
    "residual_after_attention",
    "mlp_rms",
    "mlp_gate_down",
    "mlp_gate_act",
    "mlp_gate_up",
    "mlp_norm",
    "router_input",
    "router_logits",
    "routing_weights",
    "expert_slots",
    "ep_sum",
    "routed",
    "shared_gate",
    "shared_up",
    "shared_act",
    "shared_down",
    "output",
    "next_attn_rms",
    "next_attn_gate_down",
    "next_attn_gate_act",
    "next_attn_gate_up",
    "next_attention_norm",
)

# Piece value roles to region names, for the piece holding the layer's pre-attention half ("next.")
# and for the piece holding its post-attention half ("prev." plus the following layer's norm).
PRE_ATTENTION_ROLES = {
    "next.residual": "input",
    "next.attn_rms": "attn_rms",
    "next.attn_gate_down": "attn_gate_down",
    "next.attn_gate_act": "attn_gate_act",
    "next.attn_gate_up": "attn_gate_up",
    "next.attn_in": "attention_norm",
    "next.q_proj": "q_proj",
    "next.k_proj": "k_proj",
    "next.v_proj": "v_proj",
}
POST_ATTENTION_ROLES = {
    "prev.attn_gate": "attn_gate",
    "prev.xsa_gate": "xsa_gate",
    "prev.residual_after_attention": "residual_after_attention",
    "prev.mlp_rms": "mlp_rms",
    "prev.mlp_gate_down": "mlp_gate_down",
    "prev.mlp_gate_act": "mlp_gate_act",
    "prev.mlp_gate_up": "mlp_gate_up",
    "prev.mlp_in": "mlp_norm",
    "prev.router_input": "router_input",
    "prev.router_logits": "router_logits",
    "prev.routed": "routed",
    "prev.shared_gate_proj": "shared_gate",
    "prev.shared_up_proj": "shared_up",
    "prev.shared_act": "shared_act",
    "prev.shared_down_proj": "shared_down",
    "next.residual": "output",
    "next.attn_rms": "next_attn_rms",
    "next.attn_gate_down": "next_attn_gate_down",
    "next.attn_gate_act": "next_attn_gate_act",
    "next.attn_gate_up": "next_attn_gate_up",
    "next.attn_in": "next_attention_norm",
    "final_rms": "next_attn_rms",
    "final_gate_down": "next_attn_gate_down",
    "final_gate_act": "next_attn_gate_act",
    "final_gate_up": "next_attn_gate_up",
    "final_hidden": "next_attention_norm",
}


# ``trainer_side.gated_norm_regions`` keys to the regions of the following layer's input norm.
NEXT_NORM_REGIONS = {
    "rms": "next_attn_rms",
    "gate_down": "next_attn_gate_down",
    "gate_act": "next_attn_gate_act",
    "gate_up": "next_attn_gate_up",
    "out": "next_attention_norm",
}
