import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # Define convolutional layers
        self.conv_layer1 = nn.Conv1d(3, 64, 1)
        self.conv_layer2 = nn.Conv1d(64, 64, 1)
        self.conv_layer3 = nn.Conv1d(64, 128, 1)
        self.conv_layer4 = nn.Conv1d(128, 1024, 1)
        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        # Define feed forward layer
        self.MLP =  nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        points = points.transpose(1, 2)
        # Apply convolutional layers with batch normalization and ReLU activation
        output= F.relu(self.bn1(self.conv_layer1(points)))
        output= F.relu(self.bn2(self.conv_layer2(output)))
        output= F.relu(self.bn3(self.conv_layer3(output)))
        output= F.relu(self.bn4(self.conv_layer4(output)))

        # max pool
        output= torch.amax(output, dim=-1)

        output= self.MLP(output)

        return output



# ------ TO DO ------
class seg_model(nn.Module):
        def __init__(self, num_seg_classes = 6):
            super(seg_model, self).__init__()
            # Define convolutional layers
            self.conv_layer1 = nn.Conv1d(3, 64, 1)
            self.conv_layer2 = nn.Conv1d(64, 64, 1)
            self.conv_layer3 = nn.Conv1d(64, 128, 1)
            self.conv_layer4 = nn.Conv1d(128, 1024, 1)
            # Define batch normalization layers
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(1024)
            # Define the point layer MLP for segmentation
            self.point_layer_mlp = nn.Sequential(
                nn.Conv1d(1088, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, num_seg_classes, 1),
            )
            

        def forward(self, points):
            '''
            points: tensor of size (B, N, 3)
                    , where B is batch size and N is the number of points per object (N=10000 by default)
            output: tensor of size (B, N, num_seg_classes)
            '''
            N = points.shape[1]
            points = points.transpose(1, 2)
            # Apply convolutional layers with batch normalization and ReLU activation for local features
            local_feat_output = F.relu(self.bn1(self.conv_layer1(points)))
            local_feat_output = F.relu(self.bn2(self.conv_layer2(local_feat_output)))
            # Apply convolutional layers with batch normalization and ReLU activation for global features
            global_feat_output = F.relu(self.bn3(self.conv_layer3(local_feat_output)))
            global_feat_output = F.relu(self.bn4(self.conv_layer4(global_feat_output)))
            global_feat_output = torch.amax(global_feat_output, dim=-1, keepdims=True).repeat(1, 1, N)
            # Concatenate local and global features and apply point layer MLP for segmentation
            output= torch.cat((local_feat_output, global_feat_output), dim=1)
            output= self.point_layer_mlp(output).transpose(1, 2) 

            return output



