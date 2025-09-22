import torch

# 원래 체크포인트 로드
state_dict = torch.load("./weights/evaluate_X_Temu.pth", map_location="cpu")

# 매핑 정의
key_mapping = {
    "gui_embedding.class_embed.weight": "gui_embedding.type_table.weight",
    "gui_embedding.vision_embed.weight": "gui_embedding.vision_proj.weight",
    "gui_embedding.vision_embed.bias":   "gui_embedding.vision_proj.bias",
    "gui_embedding.screen_embed.weight": "gui_embedding.screen_table.weight",
    "class_idx_reconstructor.weight":    "type_reconstructor.weight",
    "class_idx_reconstructor.bias":      "type_reconstructor.bias",
}

# 변환
new_state_dict = {}
for key, value in state_dict.items():
    if key in key_mapping:
        new_key = key_mapping[key]
        print(f"Renaming {key} -> {new_key}")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 새 state_dict 저장
torch.save(new_state_dict, "./weights/evaluate_X_Temu.pth")