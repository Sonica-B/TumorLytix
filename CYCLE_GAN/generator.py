import torch 
import torch.nn as nn


class Convblock(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,use_act = True, **kwargs):
          super().__init__()

          self.conv = nn.Sequential(
               nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs)
               if down  # encoder part where we downsample 
               else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),

               nn.InstanceNorm2d(out_channels),
               nn.ReLU(inplace=True) if use_act else nn.Identity()
          )

    def forward(self,x):
         
         return self.conv(x)
    
class Residual(nn.Module):
     
    def __init__(self,channels):
          super().__init__()

          self.resblock = nn.Sequential(
               Convblock(channels,channels,kernel_size = 3, padding = 1),
               Convblock(channels,channels,use_act = False,kernel_size = 3, padding = 1)) # As per paper, last layer has no activation hance use_Activation argument was set to false

    def forward(self,x):
         return x + self.resblock(x)  # Skip connection standard to residual block



class Generator(nn.Module): # following cycle gan paper architecture
     def __init__(self,channels,num_features= 64,num_residual_blocks = 9):
          
          super().__init__()

          self.initial = nn.Sequential(
             nn.Conv2d(channels,num_features,kernel_size=7,stride =1,padding=3,padding_mode="reflect"),
             nn.ReLU(inplace=True),  
          )
          # 64 - 128 - 256 
          self.down_blocks = nn.ModuleList(
             [
                  Convblock(num_features,num_features*2,kernel_size=3,stride=2,padding=1),
                  Convblock(num_features*2,num_features*4,kernel_size=3,stride=2,padding=1),
               ] )
        
          self.residual_block = nn.Sequential(
             *[Residual(num_features*4) for _ in range(num_residual_blocks)]
               )

          self.upsample_blocks = nn.ModuleList(
             [   # 256 - 128 - 64
                  Convblock(num_features*4,num_features*2,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),  # output padding is used to adjust the output dimension to a certain dimension while upsampling using Transposed Convolution
                  Convblock(num_features*2,num_features,down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
               ] )

          self.final_layer = nn.Conv2d(num_features,3,kernel_size=7,stride =1,padding=3,padding_mode="reflect")

     def forward(self,x):
          x = self.initial(x)
          for d in self.down_blocks:
               x = d(x)
          x = self.residual_block(x)
          for u in self.upsample_blocks:
               x = u(x)

          x = self.final_layer(x)

          return torch.tanh(x)  # Normalization to [-1,1]


# def test():

#      img_chan = 3 
#      img_size = 256 

#      x = torch.randn((3,3,256,256))

#      out = Generator(img_chan)
#      # print(out(x).shape) 

# if __name__ == "__main__":
#      test()



        
        


          
