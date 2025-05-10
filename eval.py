import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class BrainModel(nn.Module):
    def __init__(self):
        super(BrainModel, self).__init__()
        
        # first conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        # second conv layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # third conv layer
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # fourth conv layer
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        
        # adaptive pooling to handle different sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # dropout to avoid overfitting
        self.fc2 = nn.Linear(1024, 1024)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # dropout to avoid overfitting
        self.fc3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()  # for binary output
    
    def forward(self, x):
        # pass through conv layers
        x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(x))))) #First block
        x = self.pool2(self.relu4(self.conv4(self.relu3(self.conv3(x))))) #second block
        x = self.pool3(self.relu5(self.conv5(x))) #thirds block
        x = self.pool4(self.relu6(self.conv6(x))) #fourth block
        
        # global pooling
        x = self.adaptive_pool(x)
        
        # flatten before fc layers
        x = x.view(x.size(0), -1)
        
        # pass through fc layers
        x = self.relu7(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu8(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x
    


# try to use GPU if available
def try_gpu():
    if torch.cuda.is_available():
        print("Found GPU for training.")
        
        # try to use the GPU
        try:
            return torch.device('cuda')
        except:
            print("Error with GPU, falling back to CPU")
            return torch.device('cpu')
    else:
        print("No GPU found. Using CPU instead.")
        return torch.device('cpu')




def load_and_predict(directory, model_file):
    """
    Loads model and predicts class labels for brain response images.
    
    The directory argument is a folder with the same structure as the provided dataset:
    /path/to/some/images
     |_ good
        |_ Good_6s_1.png
        |_ Good_6s_2.png
        |_ ...
     |_ bad
        |_ Bad_6s_1.png
        |_ Bad_6s_2.png
        |_ ...
    
    The model_file argument is a trained model file in .pth format.
    
    This function:
    1. Reads data from the provided directory
    2. Prepares the data using the same preprocessing as during training
    3. Loads the model checkpoint
    4. Gets predicted class probabilities from the model
    5. Converts probabilities to labels: 0 (int) for "bad" class, 1 (int) for "good" class
    6. Returns a dictionary mapping absolute file paths to predicted labels
    
    Returns:
        dict: Dictionary with format {'/path/to/images/good/Good_6s_1.png': 1, 
                                     '/path/to/images/bad/Bad_6s_1.png': 0, ...}
    """

    
    # try to use GPU
    device = try_gpu()

    

    # transformations for images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize to 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    # load model
    model = BrainModel()
    

    # load weights
    try:
        if device.type == 'cuda':
            state_dict = torch.load(model_file, weights_only = True)
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(model_file, map_location = "cpu", weights_only = True)
            model.load_state_dict(state_dict)
        

        # move model to device
        model = model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    


    # set to evaluation mode
    model.eval()  
    

    image_files = []
    

    # check good folder
    good_dir = os.path.join(directory, 'good')
    if os.path.exists(good_dir):
        for file in os.listdir(good_dir):
            if file.endswith('.png'):
                image_files.append(os.path.join(good_dir, file))
    

    # check bad folder
    bad_dir = os.path.join(directory, 'bad')
    if os.path.exists(bad_dir):
        for file in os.listdir(bad_dir):
            if file.endswith('.png'):
                image_files.append(os.path.join(bad_dir, file))
    

    print(f"Found {len(image_files)} images")
    

    # dictionary to store results
    labels_dict = {}
    
    # process images in small batches to save memory
    batch_size = 16 if device.type == 'cuda' else 4
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    


    with torch.no_grad():
        for i in range(num_batches):

            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]


            batch_tensors = []
            valid_paths = []
            

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(img_path)  # Only add valid paths
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
                
            # combine into batch and move to device
            batch_input = torch.stack(batch_tensors).to(device)
            
            #get model predictions
            batch_output = model(batch_input)
            
            # process results
            for j, img_path in enumerate(valid_paths):

                # get probability value
                prob = batch_output[j].item()
                
                # Convert to int label: 0 for "bad", 1 for "good"
                # Explicitly cast to int to ensure the right type

                if prob >= 0.5:
                    pred_label = int(1)  # Explicitly convert to int for "good" class
                else:
                    pred_label = int(0)  # Explicitly convert to int for "bad" class
                

                # add to results with absolute path
                # Get the absolute path exactly as required
                abs_path = os.path.abspath(img_path)
                
                # Store in dictionary with absolute path as key and int label as value
                labels_dict[abs_path] = pred_label
    
    # clean up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    

    # Return dictionary with absolute paths as keys and int labels as values
    return labels_dict


if __name__ == "__main__":

    test_dir = "topomaps"
    model_path = "model.pth"    
    results = load_and_predict(test_dir, model_path)
    print(results)

    count = 0
    counter_correct = 0
    for path, label in results.items():
        f = os.path.basename(path).find("_") 
        if f == 4:
            true_label = 1
        else:
            true_label = 0
        if label == true_label:
            counter_correct += 1

        count += 1
    print(f"Total predictions: {len(results)}")
    print("Accuracy:  ", counter_correct * 1.0 / count)
    pass

