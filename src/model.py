# import the necessary packages
from src import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

'''
Overall, our U-Net model will consist of an Encoder class and a Decoder class. The encoder will gradually reduce the 
spatial dimension to compress information. Furthermore, it will increase the number of channels, that is, the number of 
feature maps at each stage, enabling our model to capture different details or features in our image. On the other hand, 
the decoder will take the final encoder representation and gradually increase the spatial dimension and reduce the 
number of channels to finally output a segmentation mask of the same spatial dimension as the input image.

Next, we define a Block module as the building unit of our encoder and decoder architecture. It is worth noting that all 
models or model sub-parts that we define are required to inherit from the PyTorch Module class, which is the parent 
class in PyTorch for all neural network modules.

'''


class Block(Module):
    def __init__(self, inChannels, outChannels):
        """
        The function of this module is to take an input feature map with the inChannels number of channels,
        apply two convolution operations with a ReLU activation between them and return the output feature map
        with the outChannels channels.

        The “__init__” is a reserved method in python classes. It is known as a constructor in Object-Oriented terminology.
        This method when called, allows the class to initialize the attributes of the class.

        Python super()
        The super() function allows us to avoid using the base class name explicitly. In Python, the super() function is
        dynamically called as unlike other languages, as it is a dynamic language. The super() function returns an object
        that represents the parent class.
        This function can be used to enable both Single and Multiple Inheritances.
        """
        super().__init__()
        # store the convolution and RELU layers
        '''
        CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                             bias=True, padding_mode='zeros', device=None, dtype=None)
        input size (batch size, C_in, H, W) and output (batch size, C_out, H_out, W_out )
         C denotes a number of channels, H is a height of input planes in pixels, and W is width in pixels. 
        
        groups: controls the connections between inputs and outputs. in_channels and out_channels must both be divisible
        by groups. For example, At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input
        channels and producing half the output channels, and both subsequently concatenated.
        At groups= in_channels, each input channel is convolved with its own set of filters 
        (of size = out_channels/in_channels )
        '''
        self.conv1 = Conv2d(inChannels,
                            outChannels,
                            kernel_size=config.CONV_KERNEL_SIZE,
                            groups=1)  # Give me depth of input.
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels,
                            outChannels,
                            kernel_size=config.CONV_KERNEL_SIZE,
                            groups=1)

    def forward(self, x):
        # print("In forward function and shape of input X is ", x.shape)
        """
        The purpose of forward method is to define the order in which the input data passes through the various layers
        we define the forward function which takes as input our feature map x,
        applies self.conv1 => self.relu => self.conv2 sequence of operations and returns the output feature map.
        """
        '''
        T1 =self.conv1(x)
        R1= self.relu(T1)
        T2 = self.conv2(R1)

        print("Tensor shape returned by ist layer =", self.conv1(x).size())
        print("Tensor shape returned by relu 1 =", R1.size())
        print("Tensor shape returned by 2nd layer 1 =", T2.size())
        '''
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        """
        The class constructor (i.e., the __init__ method) takes as input a tuple (i.e., channels) of channel dimensions.
        Note that the first value denotes the number of channels in our input image,
        and the subsequent numbers gradually double the channel dimension.
        """
        super().__init__()
        # store the encoder blocks and maxpooling layer
        '''
        We start by initializing a list of blocks for the encoder (i.e., self.encBlocks) with the help of 
        PyTorch’s ModuleList functionality
        Each Block takes the input channels of the previous block and doubles the channels in the output feature map. 
        '''

        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])

        # print("self.encBlock =", self.encBlocks)

        '''
        We also initialize a MaxPool2d() layer, which reduces the spatial dimension (i.e., height and width) of 
        the feature maps by a factor of 2.
        '''
        # nn.MaxPool2d is a max-pooling layer that just requires the kernel size and the stride
        # CLASS torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,return_indices=False,ceil_mode=False)
        # stride – the stride of the window. Default value is kernel_size
        self.pool = MaxPool2d(kernel_size=config.POOL_KERNEL_SIZE)

    def forward(self, x):
        """
        The function takes as input an image x , we initialize an empty blockOutputs list, storing the
        intermediate outputs from the blocks of our encoder. Note that this will enable us to later pass these
        outputs to that decoder where they can be processed with the decoder feature maps.
        """
        # initialize an empty list to store the intermediate outputs
        # Since, we also need to store the outputs of the block, we store them in a list and return this list.
        blockOutputs = []

        # loop through the encoder blocks
        '''
        we loop through each block in our encoder, process the input feature map through the block, 
        and add the output of the block to our blockOutputs list.
        '''
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store the outputs,
            # and then apply maxpooling on the output
            # print("Tensor shape provided to forward of Encoder =", x.size())
            x = block(x)
            # print("Tensor shape returned by block(x) = ",x.size())
            blockOutputs.append(x)
            # print(f"blockOutputs list first item shape {blockOutputs[0].size()}")
            # We then apply the max pool operation on our block output. This is done for each block in the encoder.
            x = self.pool(x)

        # return the list containing the intermediate outputs
        return blockOutputs


'''
Now we define our Decoder class. Similar to the encoder definition, the decoder __init__ method takes as input a tuple 
(i.e., channels) of channel dimensions. Note that the difference here, when compared with the encoder side, 
is that the channels gradually decrease by a factor of 2 instead of increasing.
'''


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels

        # we define a list of upsampling blocks (i.e., self.upconvs) that use the ConvTranspose2d layer to upsample the
        # spatial dimension (i.e., height and width) of the feature maps by a factor of 2. In addition, the layer also
        # reduces the number of channels by a factor of 2.

        # ModuleList can be indexed like a regular Python list, but modules it contains are properly registered,
        # and will be visible by all Module methods.
        #                   ConvTranspose2d(self, in_channels, out_channels, kernel_size, stride)
        # print(f"channels[0] = {channels[0]}, channels[1] {channels[1]}")
        # print(f"channels[0] = {channels[1]}, channels[2] {channels[2]}")
        self.upconvs = ModuleList([ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        # print("In model.py, printing Decoder self.upconvs = ", self.upconvs)

        # Finally, we initialize a list of blocks for the decoder (i.e., self.dec_Blocks)
        # similar to that on the encoder side.
        self.dec_blocks = ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        # print("In model.py, printing Decoder self.dec_blocks = ", self.dec_blocks)

    # we define the forward function, which takes as input our feature map x and the list of intermediate outputs from
    # the encoder (i.e., encFeatures).
    def forward(self, x, encFeatures):
        #print("Up convolved input Feature map x in decoder's forward function ", x.size())
        # print("List of Intermediate outputs in decoder's forward function", len(encFeatures))
        #print("Dim of first feature map that should be fed to ist decoder block",encFeatures[0].size() )
        #print("Dim of first feature map that should be fed to ist decoder block", encFeatures[1].size())
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # First, we upsample the input to our decoder (i.e., x) by passing it through our i-th upsampling block
            # pass the inputs through the upsampler blocks
            # print("Forward of Decoder, in for loop, Tensor before upconvs[i](x) = ", x.size())
            x = self.upconvs[i](x)
            # print("Forward of Decoder, in for loop, Tensor after upconvs[i](x) i.e upsampler blocks ", x.size())

            # Since we have to concatenate (along the channel dimension) the i-th intermediate feature map from the
            # encoder (i.e., encFeatures[i]) with our current output x from the upsampling block, we need to ensure
            # that the spatial dimensions of encFeatures[i] and x match. To accomplish this, crop the current features
            # from the encoder blocks, concatenate them with the current upsampled features, and pass the concatenated
            # output through the current decoder block
            encFeat = self.crop(encFeatures[i], x)

            # Next, we concatenate our cropped encoder feature maps (i.e., encFeat) with our current upsampled feature
            # map x, along the channel dimension
            x = torch.cat([x, encFeat], dim=1)

            # Finally, we pass the concatenated output through our i-th decoder block
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    #  we define our crop function which takes an intermediate feature map from the encoder (i.e., encFeatures) and a
    #  feature map output from the decoder (i.e., x) and spatially crops the former to the dimension of the latter.
    def crop(self, encFeatures, x):
        # To do this, we first grab the spatial dimensions of x (i.e., height H and width W).
        # grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures


# Now that we have defined the sub-modules that make up our U-Net model, we are ready to build our U-Net model class.
class UNet(Module):
    """
    It takes the following parameters as input:

    encChannels: The tuple defines the gradual increase in channel dimension as our input passes through the encoder.
                We start with 3 channels (i.e., RGB) and subsequently double the number of channels.
    decChannels: The tuple defines the gradual decrease in channel dimension as our input passes through the decoder.
                 We reduce the channels by a factor of 2 at every step.
    nbClasses: This defines the number of segmentation classes where we have to classify each pixel. This usually
               corresponds to the number of channels in our output segmentation map,
               where we have one channel for each class.
    ##################################################
    Since we are working with two classes (i.e., binary classification), we keep a single channel and use thresholding
    for classification, as we will discuss later.
    ##################################################
    retainDim: This indicates whether we want to retain the original output dimension.
    outSize: This determines the spatial dimensions of the output segmentation map. We set this to the same dimension
             as our input image (i.e., (config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)).
    """

    def __init__(self,
                 encChannels=(3, 16, 32, 64),
                 decChannels=(64, 32, 16),
                 nbClasses=config.NUM_CLASSES,
                 retainDim=True,
                 outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()

        # we initialize our encoder and decoder networks.
        #
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # we initialize a convolution head through which will later take our decoder output as input and output our
        # segmentation map with nbClasses number of channels.
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # We begin by passing our input x through the encoder. This outputs the list of encoder feature maps
        # (i.e., encFeatures). Note that the encFeatures list contains all the feature maps starting from the first
        # encoder block output to the last, as discussed previously. Therefore, we can reverse the order of feature maps
        # in this list: encFeatures[::-1].
        # grab the features from the encoder
        # print("data type returnrd by forwar_call of module.py =", type(x))
        # print("Tensor shape received at forward of model init =", x.size())
        encFeatures = self.encoder(x)
        # print("Length of list returned by self.encoder i.e encFeatures =", len(encFeatures))
        # print("encFeatures = ", encFeatures)
        # Now the encFeatures[::-1] list contains the feature map outputs in reverse order (i.e., from the last to the
        # first encoder block). Note that this is important since, on the decoder side, we will be utilizing the encoder
        # feature maps starting from the last encoder block output to the first.

        # Next, we pass the output of the final encoder block (i.e., encFeatures[::-1][0]) and the feature map outputs
        # of all intermediate encoder blocks (i.e., encFeatures[::-1][1:]) to the decoder. The output of the decoder is
        # stored as decFeatures.
        # pass the encoder features through decoder making sure that their dimensions are suited for concatenation
        # print(" tensor encFeatures[::-1][0] ",encFeatures[::-1][0].size())
        # print("List encFeatures[::-1][1:] ", len(encFeatures[::-1][1:]))
        # print("List encFeatures[::-1][1:] ", encFeatures[::-1][1:])
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:]
                                   )
        # We pass the decoder output to our convolution head to obtain the segmentation mask.
        # pass the decoder features through the regression head to obtain the segmentation mask
        map = self.head(decFeatures)

        # Finally, we check if the self.retainDim attribute is True. If yes, we interpolate the final segmentation map
        # to the output size defined by self.outSize.
        # check to see if we are retaining the original output dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)

        # return the segmentation map
        return map


# This completes the implementation of our U-Net model. Next, we will look at the training procedure for our segmentation pipeline.


############### Lets test the module #####################

if __name__ == '__main__':
    '''
    enc_block = Block(1, 64)
    x = torch.randn(1, 1, 572, 572)
    print("In main of model.py, Block object returned = ", enc_block(x).shape)
    '''
    '''
    encoder = Encoder()
    # input image
    x = torch.randn(1, 3, 572, 572)
    ftrs = encoder(x)
    for ftr in ftrs: print("In main of model.py, Encoder object returned feature maps = ", ftr.shape)
    #Out from above for loop, change in input tensor down the encoder path
    #   In main of model.py, Encoder object returned feature maps =  torch.Size([1, 16, 568, 568])
    #   In main of model.py, Encoder object returned feature maps =  torch.Size([1, 32, 280, 280])
    #   In main of model.py, Encoder object returned feature maps =  torch.Size([1, 64, 136, 136])

    # Decoder of a U-Net. Let’s make sure this implementation  works:
    decoder = Decoder()  # Use default arguments i.e channel tuple
    # provide tensor of shape attained at the end of encoder path, this will be just upconvolved in decoder
    x = torch.randn(1, 64, 136, 136)
    decFeatures= decoder(x, ftrs[::-1][1:])
    # The feature map from last encoder stage is never really concatenated to the input of ist decoder stage in fact it
    # undergoes only an “up-convolution” operation and act as input to 1st Decoder block and accepts the feature map
    # from the 2nd last position Encoder block. Similarly, the 2nd Decoder block accepts the inputs from the 3rd last
    # position Encoder block and so on. Therefore, the encoder_features are reversed before passing them to the Decoder
    # and since the feature map of last encoder stage is not concatenated to the Decoder blocks, it is not passed.
    # Hence, the input to the decoder is ftrs[::-1][1:].
    print("Encoder object returned feature maps ", decFeatures.size())

    # >> (torch.Size([1, 64, 388, 388])
    '''
