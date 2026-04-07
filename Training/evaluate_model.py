import torch
import os
import sys
import argparse

# Add path for personnal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model.cnn_architecture import FaceCNN
from backend.model.utils import DataManager

def main():
    parser = argparse.ArgumentParser(description='Evaluate Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of pictures')
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'models', 'saved_models', 'best_model.pth'), help='Path to saved model')
    parser.add_argument('--selected_persons', nargs='+', default=None, help='List of specific persons to evaluate on')
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    num_classes = len(checkpoint['class_to_idx'])
    model = FaceCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Model loaded with {num_classes} classes")

    # Load data (same split)
    data_manager = DataManager(args.data_dir, selected_persons=args.selected_persons)
    train_loader, val_loader, _ = data_manager.get_dataloaders()

    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in class_to_idx}
    class_total = {cls: 0 for cls in class_to_idx}

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[idx_to_class[label]] += 1
                if label == pred:
                    class_correct[idx_to_class[label]] += 1

    accuracy = 100 * correct / total
    print(f"Overall Accuracy: {accuracy:.2f}%")

    print("\nAccuracy per class:")
    for cls in sorted(class_to_idx.keys()):
        if class_total[cls] > 0:
            acc = 100 * class_correct[cls] / class_total[cls]
            print(f"{cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")

if __name__ == '__main__':
    main()
