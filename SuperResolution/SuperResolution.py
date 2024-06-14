import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
from PIL import Image, ImageTk
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'


t1 = v2.Compose([v2.Resize((16, 16)),
                 v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True)])

t2 = v2.Compose([v2.Resize((32, 32)),
                 v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True),])




class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, t1, t2):
        self.t1 = t1
        self.t2 = t2

        nested = os.listdir(path)
        self.items = []
        for nest in nested:
            try:
                self.items += [os.path.join(path, nest, file) for file in os.listdir(os.path.join(path, nest))]

            except NotADirectoryError:
                continue

        self.len = len(self.items)

    def __getitem__(self, index):
        name = self.items[index]
        img = Image.open(name).convert('RGB')

        x = self.t1(img)
        y = self.t2(img)

        return x, y

    def __len__(self):
        return self.len
        
        

test = './test/'
train = './train/'

batch_size = 18432

# train_loader = torch.utils.data.DataLoader(Dataset(train, t1, t2), batch_size = batch_size)
# test_loader = torch.utils.data.DataLoader(Dataset(test, t1, t2), batch_size = batch_size)
# val_loader = torch.utils.data.DataLoader(Dataset(test, t1, t2), batch_size = batch_size)




class SuperResModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.extraction = nn.Sequential(nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 3),
                                        nn.Conv2d(64, 128, kernel_size = 2, stride = 2, padding = 1),
                                        nn.Conv2d(128, 256, kernel_size = 2, stride = 2, padding = 1),)

        self.upsample = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 1),
                                      nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 1),
                                      nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 3),)

        self.raise_res = nn.Sequential(nn.ConvTranspose2d(3, 3, kernel_size = 3, stride = 2),
                                       nn.Conv2d(3, 3, kernel_size = 2, stride = 1))

    def forward(self, x):
        extracted = self.extraction(x)
        activated = nn.ReLU()(extracted)
        upsampled = (self.upsample(extracted) * 0.5 + x * 400) / 400.5
        return self.raise_res(upsampled)


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    min_loss = float('inf')
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)

        if training_loss < min_loss:
            min_loss = training_loss
            new_epoch = epoch

            with open('epoch.txt', 'w') as file:
                file.write(str(epoch))
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        
        torch.save({'Model': SuperRes_model.state_dict(), 'Optimizer': optimizer.state_dict()}, os.path.join('Checkpoints', 'SuperResModelEpoch' + str(epoch) + '.pth'))
        

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss,
        valid_loss))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


SuperRes_model = SuperResModel()
print(count_parameters(SuperRes_model))
SuperRes_model.to(device)
optimizer = optim.Adam(SuperRes_model.parameters(), lr=0.01)


SuperRes_model.load_state_dict(torch.load('SuperResModel.pth')['Model'])
optimizer.load_state_dict(torch.load('SuperResModel.pth')['Optimizer'])
# train(SuperRes_model, optimizer,torch.nn.MSELoss(), train_loader,val_loader, epochs=300, device=device)
torch.save({'Model': SuperRes_model.state_dict(), 'Optimizer': optimizer.state_dict()}, 'SuperResModel.pth')

SuperRes_model.load_state_dict(torch.load('SuperResModel.pth')['Model'])

t = v2.Compose([v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

import tkinter
from tkinter import Tk, Button, Label, Canvas, filedialog

t1 = v2.Compose([v2.Resize(128),
                 v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True)])

t2 = v2.Compose([v2.Resize(256),
                 v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True)])

class ImageEditorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Editor")

        self.image = None

        # Button to open file dialog
        self.open_button = Button(master, text="Open Image", command=self.open_image)
        self.open_button.pack()

        # Canvas to display the image
        self.canvas = Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.canvas2 = Canvas(master, width=800, height=800)
        self.canvas2.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = v2.ToPILImage('RGB')(t1(Image.open(file_path).convert('RGB')))
            photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo  # Retain reference to prevent garbage collection
            self.master.update()
            tensor = t2(self.image).to(device).unsqueeze(0)
            new_image = v2.ToPILImage('RGB')(SuperRes_model(tensor).squeeze(0))
            self.image.show()

            

            print('OTHER')
            new_image.save(file_path.split('.')[0] + '_generated.png')
            # os.startfile(file_path.split('.')[0] + '_generated.png')
            new_image.show()
            self.canvas.create_image(0, 0, anchor="nw", image=ImageTk.PhotoImage(new_image))
            self.canvas.image = photo  # Retain reference to prevent garbage collection
            self.master.update()
            
            
            # Add additional functionality here, like image editing options

def call(file_path):
    image = v2.ToPILImage('RGB')(t1(Image.open(file_path)))
    image.show()

    tensor = t2(self.image).to(device).unsqueeze(0)
    new_image = v2.ToPILImage('RGB')(SuperRes_model(tensor).squeeze(0))

    new_image.show()
    
    

if __name__ == "__main__":
    root = Tk()
    app = ImageEditorApp(root)
    root.mainloop()
    torch.cuda.empty_cache()

