# Training Our Segmentation Model
"""
Now that we have implemented our dataset class and model architecture, we are ready to construct and train our
segmentation pipeline in PyTorch.

Specifically, we will be looking at the following in detail:

   Structuring the data-loading pipeline
   Initializing the model and training parameters
   Defining the training loop
   Visualizing the training and test loss curves

   Training is probably the trickiest part of the code. Let's see what the code does:

        We start by iterating through the number of epochs, and then the batches in our training data
        We convert the images and the labels according to the device we are using, i.e., GPU or CPU
        In the forward pass we make predictions using our model and calculate loss based on those predictions and our
        actual labels. Next, we do the backward pass where we actually update our weights to improve our model
        We then set the gradients to zero before every update using optimizer.zero_grad() function
        Then, we calculate the new gradients using the loss.backward() function
        And finally, we update the weights with the optimizer.step() function
"""
# USAGE
# python train.py
if __name__ == "__main__":
    #import torch.multiprocessing as mp
    #mp.use_start_method('spawn', force=True)
    # import the necessary packages
    import optim as optim

    from src.dataset import SegmentationDataset
    from src.model import UNet
    from src import config

    from torchsummary import summary
    from torch.nn import CrossEntropyLoss  # generally used for segmentation of multiple categories.

    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    # we import other useful packages for handling our file system, keeping track of progress during training,
    # timing our training process, and plotting loss curves
    from imutils import paths
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torch
    import time
    import os

    import torch
    import torchvision
    from PIL import Image
    #import torchvision.transforms as T
    import math


    # Once we have imported all necessary packages, we will load our data and structure the data loading pipeline.

    # we first define two lists (i.e., imagePaths and maskPaths) that store the paths of all images and
    # their corresponding segmentation masks, respectively.
    # load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
    print(f"length of imagePaths = {len(imagePaths)}")
    print(f"length of maskPaths = {len(maskPaths)}")

    # partition the data into training and testing splits using 85% of the data for training and the
    # remaining 15% for testing
    # Note that this function takes as input a sequence of lists (here, imagePaths and maskPaths) and simultaneously
    # returns the training and test set images and corresponding training and test set masks which we unpack
    # on next command.
    split = train_test_split(imagePaths,
                             maskPaths,
                             test_size=config.TEST_SPLIT,
                             random_state=42
                             )
    # print("split =",split)

    # unpack the data split
    (trainImages, testImages) = split[:2]  # list_name[start:stop:steps], here 0 t0 1 not including 2
    (trainMasks, testMasks) = split[2:]  # 2 to end
    ################### For Debugging period use reduced data ############################
    trainImages = trainImages[0:(math.floor(len(trainImages)*config.PERCENT_DS))]
    testImages = testImages[0:(math.floor(len(testImages)*config.PERCENT_DS))]
    trainMasks = trainMasks[0:(math.floor(len(trainMasks)*config.PERCENT_DS))]
    testMasks = testMasks[0:(math.floor(len(testMasks)*config.PERCENT_DS))]

    ############  Do something to valudate training for hyper tuning also , also chk cross validation ##########
    ############################################################################################################

    # write the testing image paths to disk so that we can use them when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # Finally, we pass the train and test images and corresponding masks to our custom SegmentationDataset to create the
    # training dataset (i.e., trainDS) and test dataset (i.e., testDS).
    # Note that we can simply pass the transforms defined above to our custom PyTorch dataset to apply these
    # transformations while loading the images automatically.

    # create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages,
                                  maskPaths=trainMasks,
                                  transforms=config.TRANSFORMS
                                  )
    testDS = SegmentationDataset(imagePaths=testImages,
                                 maskPaths=testMasks,
                                 transforms=config.TRANSFORMS
                                 )

    # We can now print the number of samples in trainDS and testDS with the help of the len() method.
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # we create our training dataloader (i.e., trainLoader) and test dataloader (i.e., testLoader) directly by passing
    # our train dataset and test dataset to the Pytorch DataLoader class.
    # We keep the shuffle parameter True in the train dataloader since we want samples from all classes to be uniformly
    # present in a batch which is important for optimal learning and convergence of batch gradient-based optimization
    # approaches

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=config.NUM_WORKERS)
    testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                            num_workers=config.NUM_WORKERS)

    # Now that we have structured and defined our data loading pipeline,
    # we will initialize our U-Net model and the training parameters.

    # We start by defining our UNet() model. Note that the to() function takes as input our config.DEVICE
    # and registers our model and its parameters on the device mentioned.
    # initialize our UNet model
    unet = UNet().to(config.DEVICE)
    # All Layer (type), Output Shape and Params
    summary(unet, input_size=(config.NUM_CHANNELS, config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))

    # we define our loss function and optimizer, which we will use to train our segmentation model. The Adam optimizer
    # class takes as input the parameters of our model (i.e., unet.parameters()) and the learning rate
    # (i.e., config.INIT_LR) we will be using to train our model.

    # initialize loss function and optimizer
    #CrossEntropyLoss(weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean',label_smoothing=0.0)
    #the optional argument weight: should be a 1D Tensor assigning weight to each of the classes.
    #   This is particularly useful when you have an unbalanced training set.

    ########## Do somethinbg for unbalanced DS ########################################################################
    ###################################################################################################################

    # The input is expected to contain raw, unnormalized scores for each class. input has to be a Tensor of size (C)
    # for unbatched input, (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K) with K≥1 for the K-dimensional case. The
    # last being useful for higher dimension inputs, such as computing cross entropy loss per-pixel for 2D images
    lossFunc = CrossEntropyLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # We then define the number of steps required to iterate over our entire train and test set, that is, trainSteps and
    # testSteps. Given that the dataloader provides our model config.BATCH_SIZE number of samples to process at a time,
    # the number of steps required to iterate over the entire dataset (i.e., train or test set) can be calculated by
    # dividing the total samples in the dataset by the batch size.

    # calculate steps per epoch for training and test set  (i.e No of mini batches)
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    # We also create an empty dictionary, H, that we will use to keep track of our training and test loss history.
    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}


    ###############   Accuracies may also be added to dict H ########################################################

    # Finally, we are in good shape to start understanding our training loop.
    # loop over epochs
    print("[INFO] training the network...")

    # To time our training process, we use the time() function. This function outputs the time when it is called.
    # Thus, we can call it once at the start and once at the end of our training process and subtract the two outputs to
    # get the time elapsed.
    startTime = time.time()

    ################   What about the time the system sleeps ???????????  ###########################################

    # We iterate for config.NUM_EPOCHS in the training loop. Before we start training, it is important to set our model
    # to train mode, as we see on next Line. This directs the PyTorch engine to track our computations and gradients
    # and build a computational graph to backpropagate later.

    for e in tqdm(range(config.NUM_EPOCHS), desc="Epoch progress ..."):
        # set the model in training mode
        unet.train()

        # We initialize variables totalTrainLoss and totalTestLoss to track our losses in the given epoch.
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        '''
        Next, we iterate over our trainLoader dataloader, which provides a batch of samples at a time. The training loop, 
        as shown as the following for loop, comprises of the following steps:
        
           First, on first statement of for loop, we move our data samples (i.e., x and y) to the device we are 
           training our model on, defined by config.DEVICE
           
           We then pass our input image sample x through our unet model on 2nd statement and get the output prediction 
           (i.e., pred)
           On third Line, we compute the loss between the model prediction, pred and our ground-truth label y
           On Lines starting at "opt.zero_grad()", we backpropagate our loss through the model and update the parameters
           This is executed with the help of three simple steps;
                we start by clearing all accumulated gradients from previous steps on Line "opt.zero_grad()". 
                Next, we call the backward method on our computed loss function as shown on Line "loss.backward()". This
                directs PyTorch to compute gradients of our loss w.r.t. all variables involved in the computation graph.
                Finally, we call opt.step() to update our model parameters.
           In the end, Line "totalTrainLoss += loss" enables us to keep track of our training loss by adding the loss 
           for the step to the totalTrainLoss variable, which accumulates the training loss for all samples.
        This process is repeated until iterated through all dataset samples once (i.e., completed one epoch).
        '''
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            #print("\n[train.py]: Shape of img provided by DL in Train loop = ", x.size())
            #print("[train.py]: Shape of mask provided by DL in Train loop ", y.size())
            '''
            enumerate(iterable, start=0)
            Return an enumerate object. iterable must be a sequence, an iterator, or some other object which 
            supports iteration. The __next__() method of the iterator returned by enumerate() returns a tuple 
            containing a count (from start which defaults to 0) and the values obtained from iterating over iterable.
            '''
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE)) # x:input image sample, y:ground-truth label
            #print("[train.py, ]:Shape of tensor that was assigned to DEVICE, X=", x.size())
            #print("[train.py, ]:SShape of tensor y that was assigned to DEVICE", y.size())
            #print(" Normalized mask tensor y", y)
            # perform a forward pass and calculate the training loss
            pred = unet(x)

            ##################
            n, c, h, w = pred.size()    #batch size, no of classes, d1, d2
            #nt, ct, ht, wt = y.size()
            nt, ht, wt = y.size()
            #print("[train.py, Training Loop ]: Pred :BS, cl, h, w",n, c, h, w)
            #print("[train.py, Training Loop ]: Label: BS, cl, ht, wt", nt, ct, ht, wt)
            #print("[train.py, Training Loop ]: Label: BS, cl, ht, wt", nt, ht, wt)
            #print(f"[train.py, Training Loop ]: Pred type = {type(pred)} and datatype of pred = {pred.dtype}")
            #print(f"[train.py, Training Loop ]: Label: type = {type(y)} and datatype = {y.dtype}")
            #print(f"Max value of predicted output = {torch.max(pred)}, and min = {torch.min(pred)}")
            '''
            y1 = torch.squeeze(y,1) # reduce dimension telling channels of mask
            print("[train.py, Training Loop ]: dimensions of y1: ", y1.size())
            #print("Label ch 0 contents : ", y1[0][0])
            #print("Label ch 1 contents : ", y1[0][1])
            #print("Label ch 2 contents : ", y1[0][2])
            print("[train.y : training loop ]: type of Label y1[0]] : ", type(y1[0]))
            print("[train.y : training loop ]: Label dimensions y1[0] : ", y1[0].size())
            print("[train.y : training loop ]: mask unique values =", torch.unique(y1[0]))
            print("[train.y : training loop ]: Labels total non zero values =", torch.count_nonzero(y1[0]))
            #print("Frequency of each value of all values =", f = y1:eq(2):sum())
            '''
            ###############
            # Dim of input required: N, C, d1, d2 i.e batchsize, num of classes, dimensions
            # Dim of target required: N, d1, d2 i.e batchsize, dimensions
            # Output: If reduction is ‘none’, same shape as the target. Otherwise, scalar. By default it is "none"
            loss = lossFunc(pred, y)
            print(f" Training loss ={loss} i = {i}")
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far i.e batchloss++
            totalTrainLoss += loss
            print(f" totalTrainLoss ={totalTrainLoss} i = {i}")
        '''
        Once we have processed our entire training set, we would want to evaluate our model on the test set. 
        This is helpful since it allows us to monitor the test loss and ensure that our model is not overfitting to the 
        training set.
    
        While evaluating our model on the test set, we do not track gradients since we will not be learning or 
        backpropagating. Thus we can switch off the gradient computation with the help of torch.no_grad() and 
        freeze the model weights. This directs the PyTorch engine not to calculate and save gradients, 
        saving memory and compute during evaluation.
        '''
        # switch off autograd (freeze the model weights)
        with torch.no_grad():
            '''
            We set our model to evaluation mode by calling the eval() function. Then, we iterate through the test set 
            samples and compute the predictions of our model on test data. The test loss is then added to the 
            totalTestLoss, which accumulates the test loss for the entire test set.
            '''
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                '''
                #print(f"[train.py, Validation loop]: shape of pred ={pred.size()}, shape of target = {y.size()}")
                #print(f"[train.py, Validation loop]: datatype of pred ={pred.dtype}, and datatype of target = {y.dtype}")
                #print(f"pred = {pred}") # contains real numbers e.g -1082.1537, 6432.5381
                ######### Changing scalar to WxH
                pred = torch.argmax(pred, dim=1)
                ############
                #print(f"pred shape after selecting most probable class = {pred.size()}")
                #print(f"pred after selecting most probable class = {pred}")
                #print(f"[train.py, After dim change ]: Pred type = {type(pred)} and datatype of pred = {pred.dtype}")
                #print(f"[train.py, After dim change ]: Label: type = {type(y)} and datatype = {y.dtype}")
                #print(f"\ny = {y}")
                '''
                '''
                y1 = torch.squeeze(y, 1)  # reduce dimension telling channels of mask
                # use y1 instead of y to eleminate RuntimeError: only batches of spatial targets supported (3D tensors)
                # but got targets of size: : [5, 1, 512, 512]
                '''
                '''
                pred=pred.to(torch.float)
                y = y.to(torch.float)
                #print(f"[train.py, after casting ]: Pred type = {type(pred)} and datatype of pred = {pred.dtype}")
                #print(f"[train.py, after casting ]: Label: type = {type(y)} and datatype = {y.dtype}")
                
                print(f"Unique contents of pred ={torch.unique(pred)}")
                print(f"Unique contents of target/label ={torch.unique(y)}")
                '''
                val_loss = lossFunc(pred, y) # converted fron long to float to cope RuntimeError: Expected floating point type for target with class probabilities, got Long
                print(f"[train.py, after casting ]: val_loss type = {type(val_loss)} and datatype of val_loss = {val_loss.dtype}")
                print(f"[train.py, after casting ]: val_loss  = {val_loss}")
                totalTestLoss += val_loss
                print(f"[train.py, Validation loop]: shape of totalTestLoss ={totalTestLoss.size()}")

        '''
        We then obtain the average training loss and test loss over all steps, that is, avgTrainLoss and avgTestLoss on 
        Lines 120 and 121, and store them on Lines 124 and 125, to our dictionary, H, 
        which we had created in the beginning to keep track of our losses.
        '''
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        '''
        Finally, we print the current epoch statistics, including train and test losses on Lines 128-130. 
        This brings us to the end of one epoch, consisting of one full cycle of training on our train set and evaluation 
        on our test set. This entire process is repeated config.NUM_EPOCHS times until our model converges.
        '''
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

    '''
    On Lines 133 and 134, we note the end time of our training loop and subtract endTime from startTime 
    (which we had initialized at the beginning of training) to get the total time elapsed during our network training.
    '''
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    '''
    Next, we use the pyplot package of matplotlib to visualize and save our training and test loss curves on Lines 138-146. 
    '''
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    # We can do this by simply passing the train_loss and test_loss keys of our loss history dictionary, H, to the
    # plot function as shown on Lines 140 and 141.
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")

    # Finally, we set the title and legends of our plots (Lines 142-145) and save our visualizations on Line 146.
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)

    # Finally, we save the weights of our trained U-Net model with the help of the torch.save() function,
    # which takes our trained unet model and the config.MODEL_PATH as input where we want our model to be saved.
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)
'''
The loss should decrease with more and more epochs. This is a good sign. But if it is fluctuating at the end, 
this could mean the model is overfitting or that the batch_size is small. 
'''

'''
Once our model is trained, we will see a loss trajectory plot similar to the one shown in Figure 4. 
Notice that train_loss gradually reduces over epochs and slowly converges. 
Furthermore, we see that test_loss also consistently reduces with train_loss following similar trend and values, 
implying our model generalizes well and is not overfitting to the training set.
'''
'''
####################### https://medium.com/@mhamdaan/multi-class-semantic-segmentation-with-u-net-pytorch-ee81a66bba89
# Next, we initialize our model and loss function.
# We use Adam as our optimizer and Cross-Entropy Loss as our loss function.
###############################
optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)

# Ignore index is set to 255, because when executing the createTrainIdLabelImgs.py, the classes that were supposed
# to be ignored were set to 255.
loss_function = nn.CrossEntropyLoss(ignore_index=255)
'''
'''
Machines learn by means of a loss function. It’s a method of evaluating how well specific algorithm models 
the given data. If predictions deviates too much from actual results, loss function would cough up a very 
large number. 
Gradually, with the help of some optimization function, loss function learns to reduce the error in prediction. 


Then we write the logic for our train function which executes over every batch of data. 
The code is pretty straightforward. Note that while training, we don’t explicitly apply softmax on the 
predictions as the Cross-Entropy Loss function in PyTorch does that for us. 
The train function below is called for every epoch.
'''

'''
def train_function(data, model, optimizer, loss_fn, device):
    print('Entering into train function')
    loss_values = []
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)

        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
'''

#####################################


############### Lets test the module #####################
'''
The main function in Python acts as the point of execution for any program. Defining the main function in Python 
programming is a necessity to start the execution of the program as it gets executed only when the program is run 
directly and not executed when imported as a module.

In Python, it is not necessary to define the main function every time you write a program. This is because the Python 
interpreter executes from the top of the file unless a specific function is defined. Hence, having a defined starting 
point for the execution of your Python program is useful to better understand how your program works.
'''
