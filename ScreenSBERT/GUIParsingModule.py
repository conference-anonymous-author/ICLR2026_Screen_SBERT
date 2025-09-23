from ultralytics import YOLO
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.ops import roi_align
import shutil
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM, MT5Tokenizer
from paddleocr import PaddleOCR
from tqdm import tqdm

keyword2type = {
    "home": "Home", #1
    "more": "More", #2
    "find": "Search", #3
    "search": "Search",
    "profile": "Profile", #4
    "account": "Profile",
    "you": "Profile",
    "user": "Profile",
    "back": "Back", #5
    "like": "Like", #6
    "heart": "Like",
    "notification": "Notifications", #7
    "norton": "Notifications",
    "setting": "Settings", #8
    "share": "Share", #9
    "follow": "Follow", #10
    "subscri": "Follow",
    "camera": "Camera", #11
    "music": "Music", #12
    "close": "Close", #13
    "add": "Add", #14
    "chat": "Chat", #15
    "info": "Info", #16
    "help": "Help", #17
    "cart": "Cart", #18
    "next": "Next", #19
    "send": "Send", #20
    "triangle": "Send",
    "checkbox": "Checkbox", #21
    "delete": "Delete", #22
    "comment": "Comment", #23
    "message": "Message", #24
    "mail": "Message",
    "locat": "Location", #25
    "play": "Video", #26
    "DVD": "Video", 
    "bookmark": "Bookmark", #27
}

functional_types_dict = {
    'Home': 0,
    'More': 1,
    'Search': 2,
    'Profile': 3,
    'Back': 4,
    'Like': 5,
    'Notifications': 6,
    'Settings': 7,
    'Share': 8,
    'Follow': 9,
    'Camera': 10,
    'Music': 11,
    'Close': 12,
    'Add': 13,
    'Chat': 14,
    'Info': 15,
    'Help': 16,
    'Cart': 17,
    'Next': 18,
    'Send': 19,
    'Checkbox': 20,
    'Delete': 21,
    'Comment': 22,
    'Message': 23,
    'Location': 24,
    'Video': 25,
    'Bookmark': 26,
    'Other': 27
 }

def xyxy2xywh(xyxy):
    xywh = np.zeros(xyxy.shape)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2])/2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3])/2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    return xywh

def xywh2xyxy(xywh):
    xyxy = np.zeros(xywh.shape)
    xyxy[:, 0] = xywh[:, 0] - (xywh[:, 2]/2)
    xyxy[:, 1] = xywh[:, 1] - (xywh[:, 3]/2)
    xyxy[:, 2] = xywh[:, 0] + (xywh[:, 2]/2)
    xyxy[:, 3] = xywh[:, 1] + (xywh[:, 3]/2)

    return xyxy

def xywh2xyxywh(xywh):
    xyxywh = np.zeros((xywh.shape[0], 6))
    xyxy = xywh2xyxy(xywh)
    xyxywh[:, :4] = xyxy
    xyxywh[:, 4:] = xywh[:, 2:]

    return xyxywh

def sort_boxes(boxes, height=1.0, y_threshold=0.05):
    boxes = sorted(boxes, key=lambda box: box[1])
    
    y_threshold = y_threshold * height
    groups = []
    
    for box in boxes:
        y1 = box[1]
        placed = False
        for group in groups:
            if abs(group[0][1] - y1) < y_threshold:
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])
    
    sorted_groups = [sorted(group, key=lambda b: b[0]) for group in groups]
    
    final_sorted = [box for group in sorted_groups for box in group]

    return np.array(final_sorted)

class Detectioner:
    def __init__(self, model_path):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

        self.feature_maps = []

    def hook_fn(self, module, input, output):
        self.feature_maps.clear()
        self.feature_maps.append(output)

    def detect(self, image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        hook_handle = self.model.model.model[-2].register_forward_hook(self.hook_fn)

        with torch.no_grad():
            prediction = self.model.predict(source=image_path, conf=0.05, iou=0.05, verbose=False)
            xyxy = prediction[0].boxes.xyxy.cpu().numpy()
            xyxy[:, 0] /= w
            xyxy[:, 1] /= h
            xyxy[:, 2] /= w
            xyxy[:, 3] /= h

            mask = (xyxy[:, 1] >= 0.02) & (xyxy[:, 3] <= 0.98)
            xyxy = xyxy[mask]

            if xyxy.shape == (0,):
                return xyxy, None
            elif xyxy.ndim == 1:
                xyxy = np.expand_dims(xyxy, axis=0)

            xywh = xyxy2xywh(xyxy)
            xywh = sort_boxes(xywh)
            xyxywh = xywh2xyxywh(xywh)

            empty_mask = (xyxywh[:, 4]*w < 1) | (xyxywh[:, 5]*h < 1)
            xyxywh = xyxywh[~empty_mask]

            feature_map = self.feature_maps[0].to(self.device)
            fw = feature_map.shape[-1]
            
            roi_boxes = torch.tensor(xyxywh[:, :4]).float().to(self.device)
            roi_boxes[:, 0] *= w
            roi_boxes[:, 1] *= h
            roi_boxes[:, 2] *= w
            roi_boxes[:, 3] *= h
            
            roi_input = torch.cat([torch.zeros((roi_boxes.shape[0], 1)).to(self.device), roi_boxes], dim=1)
            roi_feats = roi_align(feature_map, roi_input, output_size=(7, 7), spatial_scale=fw/w).cpu()
            
            gui_feature_maps = nn.Flatten()(roi_feats).cpu().numpy()
            
        hook_handle.remove()
    
        return xyxywh, gui_feature_maps

class GUIParsingModule:
    def __init__(self):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # We used the YOLO weights from OmniParser and the Florence-2 weights for the implementation of this module.
        # https://github.com/microsoft/OmniParser
        yolo_path = "./weights/YOLO/model.pt"
        florence_2_path = "./weights/Florence-2"

        self.detectioner = Detectioner(yolo_path)
        self.sbert = SentenceTransformer("sentence-transformers/LaBSE", device=self.device)
        self.captioner = AutoModelForCausalLM.from_pretrained(florence_2_path, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        self.ocr = PaddleOCR(lang='en')

    def detect(self, image_path):
        return self.detectioner.detect(image_path)
    
    def crop(self, image_path, coords):
        if os.path.exists("./temp_images"):
            shutil.rmtree("./temp_images")
        os.mkdir("./temp_images")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        for idx, box in enumerate(coords):
            gui = image.crop((box[0]*w, box[1]*h, box[2]*w, box[3]*h))
            gui.save(f"./temp_images/{idx}.jpg")

    def embed_texts(self, num_gui):
        gui_text_embeds = []
        for idx in tqdm(range(num_gui), desc="OCR and Embedding Texts"):
            image = Image.open(f"./temp_images/{idx}.jpg").convert("RGB")
            image = np.asarray(image)

            ocr_results = self.ocr.predict(image)
            texts = ocr_results[0]['rec_texts']
            if len(texts) == 0:
                gui_text_embeds.append(np.zeros((768,)))
            else:
                sentence = ""
                for text in texts:
                    sentence += (text + " ")
                gui_text_embeds.append(self.sbert.encode(sentence))

        return np.array(gui_text_embeds) 
    
    def embed_types(self, num_gui):
        gui_images = []
        for idx in range(num_gui):
            image = Image.open(f"./temp_images/{idx}.jpg").convert("RGB")
            image = image.resize((64, 64))
            gui_images.append(np.asarray(image))
        gui_images = np.array(gui_images)

        with torch.no_grad():
            inputs = self.processor(
                images=gui_images, 
                text=["<CAPTION>"]*num_gui,
                return_tensors="pt", 
                padding=False,
                do_resize=False
            )

            generated_ids = self.captioner.generate(
                input_ids=inputs["input_ids"].to(self.device),
                pixel_values=inputs["pixel_values"].to(self.device),
                max_new_tokens=64,
                num_beams=1, 
                do_sample=False
            )

            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )

        # Rule-based Classification of Funtional Types
        gui_types = []
        for caption in generated_text:
            for keyword, functional_type in keyword2type.items():
                if keyword.lower() in caption.lower():
                    gui_types.append(functional_types_dict[functional_type])
                    break
            else:
                gui_types.append(functional_types_dict["Other"])

        return np.array(gui_types)
            

    def parse(self, image_path):
        gui_coords, gui_feature_maps = self.detect(image_path)
        self.crop(image_path, gui_coords)

        num_gui = gui_coords.shape[0]
        gui_text_embeds = self.embed_texts(num_gui)
        gui_types = self.embed_types(num_gui)

        return gui_coords, gui_feature_maps, gui_text_embeds, gui_types

if __name__ == "__main__":
    gui_parsing_module = GUIParsingModule()
    
    image_path = "./example_screenshot.jpg"
    gui_coords, gui_feature_maps, gui_text_embeds, gui_types = gui_parsing_module.parse(image_path=image_path)

    print(f"\nParsing results of {image_path}:")
    print(f"Coordinates: {gui_coords.shape}")
    print(f"Feature Maps: {gui_feature_maps.shape}")
    print(f"Text Embeddings: {gui_text_embeds.shape}")
    print(f"Functional Types: {gui_types.shape}")

