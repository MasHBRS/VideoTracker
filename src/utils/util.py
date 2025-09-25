"""
TODO:
1. add background extract as well to the output 
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.ops import boxes
from tqdm import tqdm
import torch.nn.functional as F
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageExtractor():
    def __init__(self,max_objects_in_scene)-> None:
        self.max_objects_in_scene=max_objects_in_scene
        pass
    def extract_masked_images(self,imgs,masks)-> torch.Tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]
        
        output=torch.zeros(size=(self.max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[max_objects_in_scene, C, H, W]
        msks=torch.zeros(size=(self.max_objects_in_scene,*masks.shape),device=imgs.device) # shape=[max_objects_in_scene, H, W]
        for uc in masks.unique():
            output[uc.item()]=imgs*(masks==uc)
            msks[uc.item()]=(masks==uc)
        return output,msks 

    def extract_bboxed_images(self,imgs,bboxes)-> torch.tensor:
        assert imgs.dim()==3 # imgs.shape=[C,H,W]

        output=torch.zeros(size=(self.max_objects_in_scene,*imgs.shape),device=imgs.device) # shape=[max_objects_in_scene, C, H, W]
        output[0]=imgs
        for idx,box in enumerate(bboxes):
            if torch.all(box == -1):
                continue
            output[idx+1,:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]=imgs[:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]
            output[0,:,int(box[1].item()):int(box[3].item()),int(box[0].item()):int(box[2].item())]=0
        return output


def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def save_model(model, optimizer, epoch, stats):
    """ Saving model checkpoint """
    
    if(not os.path.exists("checkpoints")):
        os.makedirs("checkpoints")
    savepath = f"checkpoints/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def visualize_progress(loss_iters, train_loss, val_loss, valid_acc, start=0):
    """ Visualizing loss and accuracy """
    #plt.style.use('seaborn')
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)
    
    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")
    
    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs, train_loss, c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")
    
    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")
    
    plt.show()
    return

def train_epoch(model, train_loader, optimizer, criterion, epoch, device,trainingmode=0):
    """ Training a model for one epoch """
    
    loss_list = []
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device,trainingmode=0):
    """ Evaluating the model for either validation or test """
    correct = 0
    total = 0
    loss_list = []
    
    #for images, labels in eval_loader:
    for coms,bboxs,masks,rgbs,flows in eval_loader:
        images = rgbs.to(device)
        
        # Forward pass only to get logits/output

        recons = model(images,boxes=bboxs) if trainingmode==0 else model(images,masks=masks) if trainingmode==1 else  model(images,boxes=bboxs)
                 
        loss = criterion(recons, images)
        loss_list.append(loss.item())
            
        # Get predictions from the maximum value
        correct += len( torch.where(recons==images)[0] ) # TODO: check this !
        total += len(images) # TODO: I think it should be = 1.
                 
    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    
    return accuracy, loss


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epochs, tboard=None, start_epoch=0,trainingmode=0):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []

    for epoch in range(num_epochs):
        print(f"Started Epoch {epoch+1}/{num_epochs}...")
        
        # validation epoch
        print("  --> Running valdiation epoch")
        model.eval()  # important for dropout and batch norms
        accuracy, loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device,
                    trainingmode=trainingmode)
        valid_acc.append(accuracy)
        val_loss.append(loss)
        tboard.add_scalar(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
        tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
        
        # training epoch
        print("  --> Running train epoch")
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device,
                trainingmode=trainingmode
            )
        scheduler.step()
        train_loss.append(mean_loss)
        tboard.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

        loss_iters = loss_iters + cur_loss_iters
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"    Train loss: {round(mean_loss, 5)}")
        print(f"    Valid loss: {round(loss, 5)}")
        print(f"    Valid Accuracy: {accuracy}%")
        print("\n")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters, valid_acc

def l1_and_ssim_loss_function(recons, target,lambda_l1=0.8 ,lambda_ssim=0.2):
    """
    Combined loss function for joint optimization
    """
    recons_loss = F.l1_loss(recons, target)
    #ssim_loss = 1 - ssim(recons, target)
    #loss = lambda_l1 * recons_loss + lambda_ssim * ssim_loss

    return recons_loss   