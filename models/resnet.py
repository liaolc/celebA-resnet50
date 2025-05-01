import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F 


__all__ = ["ResNet", "resnet50"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_attributes=40, zero_init_residual=False, num_classes=10, radial_radius=3, radial_decay=0.5, upper_mask_level_threshold=0.8):
        super(ResNet, self).__init__()

        # interpretability params
        self.mask_generator = MaskGenerator()
        self.classifier = Classifier(num_classes)
        self.radial_radius = radial_radius
        self.radial_decay = radial_decay
        self.upper_mask_level_threshold = upper_mask_level_threshold

        # ResNet params
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.stem = fc_block(512 * block.expansion, 512)
        self.mask_generator = MaskGenerator()
        for i in range(num_attributes):
            setattr(
                self,
                "classifier" + str(i).zfill(2),
                Classifier(num_classes=1),
            )
        self.num_attributes = num_attributes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Generate mask using the mask generator
        raw_mask = self.mask_generator(x)
        
        # Apply sigmoid to constrain values between 0 and 1
        soft_mask = torch.sigmoid(raw_mask)
        
        # Apply radial mask instead of threshold-based approach
        radial_mask = apply_radial_mask(soft_mask, radius=self.radial_radius, decay_factor=self.radial_decay)
        
        # Create binary mask for pixels above threshold
        binary_mask = (radial_mask > self.upper_mask_level_threshold).float()
        
        # Expand masks to match input dimensions
        radial_mask_expanded = radial_mask.repeat(1, 3, 1, 1)
        binary_mask_expanded = binary_mask.repeat(1, 3, 1, 1)
        
        # Apply mask to input - INVERTED: now 1 means full masking, 0 means no masking
        # For binary mask, we use it directly (1 means fully masked)
        # For radial mask, we multiply by (1 - mask_expanded)
        masked_x = x * (1 - radial_mask_expanded)
        
        # Apply binary mask on top - fully mask pixels above threshold
        masked_x = masked_x * (1 - binary_mask_expanded)
        
        # get predictions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.stem(x)

        masked_x_original = masked_x.clone()

        masked_x = self.conv1(masked_x)
        masked_x = self.bn1(masked_x)
        masked_x = self.relu(masked_x)
        masked_x = self.maxpool(masked_x)

        masked_x = self.layer1(masked_x)
        masked_x = self.layer2(masked_x)
        masked_x = self.layer3(masked_x)
        masked_x = self.layer4(masked_x)

        masked_x = self.avgpool(masked_x)
        masked_x = masked_x.view(masked_x.size(0), -1)
        masked_x = self.stem(masked_x)

        masked_logits = []
        unmasked_mlp_features = []
        masked_mlp_features = []

        masked_encoding = masked_x.clone()
        unmasked_encoding = x.clone()

        unmasked_logits = []
        for i in range(self.num_attributes):
            classifier = getattr(self, "classifier" + str(i).zfill(2))
            
            # Extract MLP features for both masked and unmasked inputs
            unmasked_mlp_features.append(self.classifier.get_mlp_features(x, layer_idx=1))
            masked_mlp_features.append(self.classifier.get_mlp_features(masked_x, layer_idx=1))

            # Get predictions
            unmasked_logits.append(classifier(x))
            masked_logits.append(classifier(masked_x))
        unmasked_logits = torch.cat(unmasked_logits, 1)
        masked_logits = torch.cat(masked_logits, 1)
        unmasked_mlp_features = torch.cat(unmasked_mlp_features, 1)
        masked_mlp_features = torch.cat(masked_mlp_features, 1)
        # return y
        return {
            'mask': soft_mask,
            'soft_mask': soft_mask,
            'radial_mask': radial_mask,
            'binary_mask': binary_mask,  # Add binary mask to outputs
            'masked_input': masked_x_original,
            'unmasked_logits': unmasked_logits,
            'masked_logits': masked_logits,
            'unmasked_mlp_features': unmasked_mlp_features,
            'masked_mlp_features': masked_mlp_features,
            'masked_encoding': masked_encoding,
            'unmasked_encoding': unmasked_encoding
        }
    


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        TO ADD: Adjustable radial_radius and radial_decay
    """
    # self, block, layers, num_attributes=40, zero_init_residual=False, 
    # num_classes=10, radial_radius=3, radial_decay=0.5, upper_mask_level_threshold=0.8
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], radial_radius=1, radial_decay=0.2, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])
    return model


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


class MaskGenerator(nn.Module):
    def __init__(self):
        super(MaskGenerator, self).__init__()
        # Enhanced U-Net architecture with more filters and layers
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4)  # Add dropout for regularization
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 = 256 (from skip) + 256 (from upconv)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 = 128 (from skip) + 128 (from upconv)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 (from skip) + 64 (from upconv)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoder path with skip connections
        upconv3 = self.upconv3(bottleneck)
        concat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        # Output mask - use clipped ReLU instead of sigmoid
        output = self.final(dec1)
        # Return raw output without clamping
        return output
    
# Function to apply radial mask to a soft mask
def apply_radial_mask(soft_mask, radius=3, decay_factor=0.2):
    """
    Apply a radial influence pattern to a soft mask.
    
    Args:
        soft_mask: Tensor of shape [B, 1, H, W] containing the original soft mask values
        radius: Integer radius of influence (in pixels)
        decay_factor: Float factor for how quickly the influence decays with distance
        
    Returns:
        Tensor of shape [B, 1, H, W] containing the radial-influenced mask
    """
    batch_size, channels, height, width = soft_mask.shape
    
    # Initialize the output tensor
    radial_mask = torch.zeros_like(soft_mask)
    
    # Create a grid of coordinates for the kernel
    y_coords = torch.arange(-radius, radius+1, device=soft_mask.device)
    x_coords = torch.arange(-radius, radius+1, device=soft_mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Calculate distances from center for the kernel
    distances = torch.sqrt(y_grid.float()**2 + x_grid.float()**2)
    
    # Create the radial kernel (1 at center, decaying with distance)
    kernel = torch.exp(-decay_factor * distances)
    kernel = kernel / kernel.sum()  # Normalize
    
    # Reshape kernel for convolution
    kernel = kernel.view(1, 1, 2*radius+1, 2*radius+1)
    
    # Apply convolution to get the radial mask
    radial_mask = F.conv2d(soft_mask, kernel, padding=radius)
    
    # Ensure values are still between 0 and 1
    radial_mask = torch.clamp(radial_mask, 0.0, 1.0)
    
    return radial_mask

# Define the Classifier Network
class Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Multi-layer MLP structure
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def get_mlp_features(self, x, layer_idx=1):  # layer_idx=1 for first MLP layer
        # x = self.features(x)
        x = torch.flatten(x, 1)
        
        # Extract features from specified MLP layer
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i == layer_idx * 2 - 1:  # ReLU after linear layer
                return x
        return x
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x
    

class MaskedLoss(nn.Module):
    def __init__(self, lambda_mask=1.0, lambda_fully_masked=0.05, lambda_smoothness=0.1, 
                 dynamic_masked_weight_min=1.0, dynamic_masked_weight_max=5.0,
                 lambda_alignment=0.5, upper_mask_level_threshold=0.8, train_sample_prob=1.0, enc_lambda_alignment=0.5):  # Add threshold parameter
        super(MaskedLoss, self).__init__()
        self.lambda_mask = lambda_mask
        self.lambda_fully_masked = lambda_fully_masked
        self.lambda_smoothness = lambda_smoothness
        self.lambda_alignment = lambda_alignment  # Weight for the feature alignment loss
        self.upper_mask_level_threshold = upper_mask_level_threshold  # Store threshold
        
        # Parameters for dynamic masked loss weighting
        self.dynamic_masked_weight_min = dynamic_masked_weight_min
        self.dynamic_masked_weight_max = dynamic_masked_weight_max

        self.enc_lambda_alignment = enc_lambda_alignment
        
        # self.mask_binauroc = BinaryAUROC().to('cuda')
        # self.unmask_binauroc = BinaryAUROC().to('cuda')

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction="mean", weight=1 / torch.sqrt(train_sample_prob)
        ).cuda()
        
    def forward(self, outputs, targets):
        soft_mask = outputs['soft_mask']
        radial_mask = outputs['radial_mask']  # Use radial mask instead of hard_mask
        binary_mask = outputs['binary_mask']  # Use binary mask from model
        mask = outputs['mask']
        unmasked_logits = outputs['unmasked_logits']
        masked_logits = outputs['masked_logits']
        masked_encoding = outputs['masked_encoding']
        unmasked_encoding = outputs['unmasked_encoding']


        # self.mask_binauroc.update(masked_logits, targets.float())
        # self.unmask_binauroc.update(unmasked_logits, targets.float())
        
        # Extract MLP features for alignment loss
        unmasked_mlp_features = outputs['unmasked_mlp_features']
        masked_mlp_features = outputs['masked_mlp_features']
        
        # 1. Calculate per-sample losses for both masked and unmasked inputs
        masked_losses = self.bce_loss(masked_logits, targets)
        unmasked_losses = self.bce_loss(unmasked_logits, targets)
        
        # Calculate the mean losses for comparison
        masked_loss_mean = masked_losses #.mean()
        unmasked_loss_mean = unmasked_losses #.mean()
        
        # Calculate the gap between masked and unmasked loss
        loss_gap = torch.clamp(masked_loss_mean - unmasked_loss_mean, min=0.0)
        
        # Compute the dynamic weight factor based on the loss gap
        dynamic_weight = torch.clamp(
            1.0 + loss_gap,
            min=self.dynamic_masked_weight_min,
            max=self.dynamic_masked_weight_max
        )
        
        # Apply the dynamic weight to the masked loss
        weighted_masked_loss = dynamic_weight * masked_loss_mean
        
        # Total classification loss combines weighted masked loss and unmasked loss
        classification_loss = weighted_masked_loss + unmasked_loss_mean
        
        # 2. Masking loss - encourage masking as many pixels as possible
        # With inverted interpretation, we want to encourage higher values (closer to 1)
        mask_mean = torch.mean(soft_mask)
        # Add exponential penalty for unmasked pixels to create stronger gradient
        # Now we want to maximize mask_mean, so we penalize (1 - mask_mean)
        masking_loss = self.lambda_mask * ((1 - mask_mean) ** 2)  # Quadratic penalty for unmasked pixels
        
        # 3. Binary loss - encourage values to be either 0 or 1 (perfect entropy)
        # This creates a strong gradient pushing values toward 0 or 1
        # The term radial_mask * (1 - radial_mask) is maximum at 0.5 and minimum at 0 and 1
        # So we want to minimize this term to push values toward 0 or 1
        # Apply binary loss to the radial mask instead of the soft mask
        binary_loss = self.lambda_fully_masked * torch.mean(radial_mask * (1 - radial_mask))
        
        # 4. Smoothness regularization - encourage spatial continuity in masks
        batch_size, _, height, width = soft_mask.size()
        
        h_grad = torch.abs(soft_mask[:, :, :, 1:] - soft_mask[:, :, :, :-1])
        v_grad = torch.abs(soft_mask[:, :, 1:, :] - soft_mask[:, :, :-1, :])
        
        smoothness_loss = (torch.sum(h_grad) + torch.sum(v_grad)) / (batch_size * height * width)
        smoothness_term = self.lambda_smoothness * smoothness_loss
        
        # 5. Feature alignment loss - now using MLP features
        # Normalize the MLP features to compute cosine similarity
        unmasked_mlp_norm = F.normalize(unmasked_mlp_features, p=2, dim=1)
        
        # Detach masked features to prevent gradients from flowing through the masked path
        masked_mlp_norm = F.normalize(masked_mlp_features.detach(), p=2, dim=1)
        
        # Compute cosine similarity between normalized MLP feature vectors
        cosine_similarity = torch.sum(unmasked_mlp_norm * masked_mlp_norm, dim=1)
        
        # Convert similarity to a divergence measure (1 - similarity) and take mean
        alignment_divergence = 1.0 - cosine_similarity.mean()
        
        # Apply weight to the alignment loss
        alignment_loss = self.lambda_alignment * alignment_divergence

        # 6. Encoding alignment loss 
        enc_alignment_similarity = torch.sum(unmasked_encoding * masked_encoding, dim=1)
        enc_alignment_divergence = 1.0 - enc_alignment_similarity.mean() 
        enc_alignment_loss = self.enc_lambda_alignment * enc_alignment_divergence
        
        # Total loss with alignment term
        total_loss = classification_loss + masking_loss + binary_loss + smoothness_term 
        
        unet_loss = enc_alignment_loss 
        resnet_loss = masked_loss_mean + unmasked_loss_mean 

        # Calculate binary mask metrics (only count fully masked pixels > threshold)
        fully_masked_pixels = (radial_mask > self.upper_mask_level_threshold).float()
        fully_masked_pct = 100 * torch.mean(fully_masked_pixels).item()
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'masked_loss': masked_loss_mean,
            'unmasked_loss': unmasked_loss_mean,
            'resnet_loss': resnet_loss,
            'unet_loss': unet_loss,
            'masking_loss': masking_loss,
            'fully_masked_loss': binary_loss,
            'smoothness_loss': smoothness_term,
            'alignment_loss': alignment_loss,  # Add the new loss to returned dict
            'alignment_divergence': alignment_divergence,  # Add raw divergence value
            'mask_mean': mask_mean,
            'binary_mask_mean': torch.mean(radial_mask).item(),  # Use radial mask instead of hard_mask
            'fully_masked_pct': fully_masked_pct,
            'dynamic_weight': dynamic_weight,
            'enc_alignment_loss': enc_alignment_loss,
            
        }
