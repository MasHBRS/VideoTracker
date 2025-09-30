import shutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src')))
from loader.Dataset import VideoDataset
from torch.utils.data import DataLoader
from models.transformers.encoders.vit_encoder import ViT
from models.transformers.decoders.vit_decoder import ViT_Decoder
from models.transformers.CustomTransformer import CustomizableTransformer
from utils.util import count_model_params, train_epoch, eval_model, train_model, load_model 
from matplotlib import pyplot as plt
from matplotlib import patches
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.loss_function import ImprovedLoss5D,ReconstructionLoss_L1_Ssim, ReconstructionLoss_PSNR_SSIM, ReconstructionLoss_MSE_SSIM
from loader.transforms import RGBNormalizer,Composition,CustomResize,RandomHorizontalFlip,RandomVerticalFlip,CustomColorJitter




general_configs={
"data_path":"/home/nfs/inf6/data/datasets/MOVi/movi_c/",
"original_number_of_frames_per_video":24,
"selected_number_of_frames_per_video":4,
"max_objects_in_scene":11,
"batch_size":64,
"device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
"img_height":64,
"img_width":64,
"channels":3,
"learning_rate":3e-4,
"num_epochs":60,
"trainingMode":1 #0 training by bounding boxes, 1 training by masks. 
}

encoder_configs={
        "token_dim":128,
        "attn_dim":128,
        "num_heads":4,
        "mlp_size":512,
        "num_tf_layers":4
}
decoder_configs={
        "token_dim":128,
        "attn_dim":128,
        "num_heads":4,
        "mlp_size":512,
        "num_tf_layers":4
}

data_transform_config={
        "img_height":general_configs["img_height"],
        "img_width":general_configs["img_width"],
        "vFlip_probability":0.3,
        "hFlip_probability":0.3,
        "color_jitter_brightness":(0.8, 1.2),
        "color_jitter_hue":(-0.3, 0.3),
        "color_jitter_contrast":(0.6, 1.8),
        "color_jitter_saturation":(0.5, 1.5)
}


transform_composition = Composition([
                                        RGBNormalizer(),
                                        CustomResize((data_transform_config["img_height"],data_transform_config["img_width"])),
                                        RandomVerticalFlip(data_transform_config["vFlip_probability"]),
                                        RandomHorizontalFlip(data_transform_config["hFlip_probability"]),
                                        #CustomColorJitter(
                                        #    brightness=data_transform_config["color_jitter_brightness"],
                                        #    hue=data_transform_config["color_jitter_hue"],
                                        #    contrast=data_transform_config["color_jitter_contrast"],
                                        #    saturation=data_transform_config["color_jitter_saturation"]
                                        #)
                                    ])
#transform_composition=None
validation_dataset = VideoDataset(data_path=general_configs["data_path"],
                            split='validation',
                            original_number_of_frames_per_video=general_configs["original_number_of_frames_per_video"],
                            selected_number_of_frames_per_video=general_configs["selected_number_of_frames_per_video"],
                            max_objects_in_scene=general_configs["max_objects_in_scene"],
                            halve_dataset=True,
                            is_test_dataset=False,
                            transforms=transform_composition)
valid_loader = DataLoader(dataset=validation_dataset,
                            batch_size=general_configs["batch_size"],
                            shuffle=False,
                            drop_last=True)
train_dataset = VideoDataset(data_path=general_configs["data_path"],
                            split='train',
                            original_number_of_frames_per_video=general_configs["original_number_of_frames_per_video"],
                            selected_number_of_frames_per_video=general_configs["selected_number_of_frames_per_video"],
                            max_objects_in_scene=general_configs["max_objects_in_scene"],
                            halve_dataset=False,
                            is_test_dataset=False,
                            transforms=transform_composition)
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=general_configs["batch_size"],
                            shuffle=True,
                            drop_last=True)


iterator=iter(train_loader)
bboxes,masks,rgbs=next(iterator)
print(f"shapes: \r\n{bboxes.shape=},\r\n{masks.shape=},\r\n{rgbs.shape=},\r\n============================================")
print(len(train_loader))


def defineVIT():
        return  ViT(
                img_height=general_configs["img_height"],
                img_width=general_configs["img_width"],
                channels=general_configs["channels"],
                max_objects_in_scene=general_configs["max_objects_in_scene"],
                frame_numbers=general_configs["selected_number_of_frames_per_video"],
                token_dim=encoder_configs["token_dim"],
                attn_dim=encoder_configs["attn_dim"],
                num_heads=encoder_configs["num_heads"],
                mlp_size=encoder_configs["mlp_size"],
                num_tf_layers=encoder_configs["num_tf_layers"]).to(general_configs["device"])

vit=defineVIT()
print(f"ViT has {count_model_params(vit)} parameters")


def defineDecoder():
    return ViT_Decoder(
    batch_size=general_configs["batch_size"],
    img_height=general_configs["img_height"],
    img_width=general_configs["img_width"],
    channels=general_configs["channels"],
    frame_numbers=general_configs["selected_number_of_frames_per_video"],
    token_dim=decoder_configs["token_dim"],
    attn_dim=decoder_configs["attn_dim"], 
    num_heads=decoder_configs["num_heads"], 
    mlp_size=decoder_configs["mlp_size"], 
    num_tf_layers=decoder_configs["num_tf_layers"],
    max_objects_in_scene=general_configs["max_objects_in_scene"],
    device=general_configs["device"]
).to(general_configs["device"])


decoder=defineDecoder()
print(f"Decoder has {count_model_params(decoder)} parameters")

transformer=CustomizableTransformer(encoder=vit, decoder=decoder).to(general_configs["device"])
assert count_model_params(decoder)+count_model_params(vit)==count_model_params(transformer)
print(f"transformer has {count_model_params(transformer)} parameters")

#criterion=ReconstructionLoss_MSE_SSIM(device=general_configs["device"],lambda_mse=1,lambda_ssim=0.01)
#criterion=ReconstructionLoss_L1_Ssim(device=general_configs["device"],lambda_l1=0.1,lambda_ssim=0.9)
criterion=ImprovedLoss5D()
optimizer = torch.optim.Adam(transformer.parameters(), lr=general_configs["learning_rate"])
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)
TBOARD_LOGS = os.path.join(os.getcwd(), "../tboard_logs", "ViT_30")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)

shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

train_loss, val_loss, loss_iters, valid_acc = train_model(
        model=transformer,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=general_configs["num_epochs"],
        tboard=writer,
        trainingmode=general_configs["trainingMode"],
        conf=(decoder_configs,encoder_configs,general_configs),saveImagesPerEachEpoch=True   )

from utils.util import save_model
stats = {
    "train_loss": train_loss,
    "valid_loss": val_loss,
    "loss_iters": loss_iters,
    "valid_acc": valid_acc
}
save_model(transformer, optimizer, epoch=general_configs["num_epochs"], stats=stats)

