import torch 
import torch.nn as nn

class Block(nn.Module):

    def __init__(self,in_channels,out_channels,stride):

        super().__init__()

        # From the Cycle GAN paper - we follow the discriminator architecture

        # "reflect" - avoids artificats during generation by mainytaining continuity near boundaries
        # Instance Norm is used to normalize each image independantly and reduce spatial dependencies between images in a abatcj to extract high level content in each image
        # Leaky RELU - better gradient stability as it avoids deactiviting neurons with -ve activation

        self.conv= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4,stride,1,bias=True,padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):

        x = self.conv(x)
        # print(f"Shape after Block: {x.shape}") 
        return x
    

class Discriminator(nn.Module):

    def __init__(self,in_channels,features=[64,128,256,512]):

        super().__init__()

        # Instance Norm is not applied after first layer because we want to detect basic features from input image first without disruption like intensity and edges and in later layers where important patterns r required to be learnt to generate images , we need to ignore the basic features to focus on more importnat features for regeneration, Hence instance norm is applied to later layers 
        self.start = nn.Sequential(
            nn.Conv2d(in_channels,features[0],kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        ) 
        # after this operation , my input channels to next layer become 64

        layers = []
        in_channels = features[0]

        for feature in features[1:]:

            layers.append(Block(in_channels,feature,stride = 2 if feature == features[-1] else 2))  # change from video (stride =1 )
            in_channels = feature
        
        # last and final layer has one single channel as it tells us whether the generated image is fake or real or quantifies 'Realness of the generated image"
        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        self.model = nn.Sequential(*layers)  # bundles the entire layer sequence into 1 for faster compute and less memory 

    def forward(self, x):
        # Starting layer
        x = self.start(x)
        # print(f"Shape after start: {x.shape}") 

        # Pass through the rest of the layers
        x = self.model(x)
        # print(f"Shape after final layer: {x.shape}") 
        
        # Final output with sigmoid
        return torch.sigmoid(x) # we need the final value to be between 0 and 1 , hence sigmoid function
    
# Following the PATCH GAN principle , we get a 15x15 output where each value has a receptive field of 70x70 in the original image
# def test():

#     x = torch.randn((5,3,256,256))
#     model = Discriminator(in_channels=3)
#     pred = model(x)
#     print (pred.shape)


# if __name__ == "__main__":
#     test()