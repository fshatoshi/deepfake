# Importation of essential libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse

# Plotting libraries
import matplotlib.pyplot as plt

# Progression bar
from tqdm import tqdm  # Correction: "from tqdm import tqdm" au lieu de "import tqdm as tqdm"

# Add path for personnal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation for personnal modules - CORRECTION ICI
from backend.model.cnn_architecture import FaceCNN  # Ajout de "backend."
from backend.model.utils import DataManager        # Ajout de "backend."

def main():
    # 1-Adding for all arguments
    parser = argparse.ArgumentParser(description='CPU Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of pictures')
    parser.add_argument('--epochs' , type=int , default=50 ,help='Number of epochs')
    parser.add_argument('--batch_size' , type=int , default=16 , help='Batch size')
    parser.add_argument('--learning_rate' , type=float , default=0.001 , help='Learning rate')
    parser.add_argument('--weight_decay' , type=float , default=1e-4 , help='L2 regularization (Adam)')
    parser.add_argument('--max_samples_per_class', type=int, default=None, help='Limit samples per class')
    parser.add_argument('--max_classes', type=int, default=None, help='Limit number of classes (persons)')
    parser.add_argument('--selected_persons', nargs='+', default=None, help='List of specific persons to train on')
    args = parser.parse_args()

    torch.manual_seed(42)

    # We needn't GPU for training
    print(f"\n Training on CPU...")
    print(f" . batch size : {args.batch_size} \n . epochs : {args.epochs} \n . learning rate : {args.learning_rate} \n . weight decay : {args.weight_decay}")
    
    # Folder's verification
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory {args.data_dir} does not exist.")
        return
    
    # Data loading
    print("\n📂 Loading data...")
    data_manager = DataManager(args.data_dir, batch_size=args.batch_size, max_samples_per_class=args.max_samples_per_class, max_classes=args.max_classes, selected_persons=args.selected_persons)
    
    # CORRECTION: La méthode s'appelle get_dataloaders() pas get_loaders()
    train_loader , val_loader , class_to_idx = data_manager.get_dataloaders()
    
    print("Dataset ")
    print (f" . classes : {len(class_to_idx)} \n . training samples : {len(train_loader.dataset)} \n . validation samples : {len(val_loader.dataset)}")
   
    # Model initialization
    print("\n🔧 Initializing model..")
    model = FaceCNN(num_classes=len(class_to_idx))

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total of parameters : {total_params:,}")  # Correction: ajout du f-string

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    # Saving
    save_dir = 'backend/models/saved_models'
    os.makedirs(save_dir, exist_ok=True)  # Ajout: créer le dossier s'il n'existe pas

    # History
    history = {
        'train_loss': [],
        'val_loss': [],      # Ajout: val_loss manquait
        'val_accuracy': []
    }
    best_acc = 0.0
    
    # Training loop
    print("\n🚀 Starting training...")
    
    for epoch in range(args.epochs):  # Correction: epochs -> epoch
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader , desc =f"Epoch {epoch+1}/{args.epochs}")
        
        for images , labels in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation part
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / total if total else 0.0
        accuracy = 100 * correct / total

        # History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)

        print(f"\n  📊 Results:")
        print(f"  • Train Loss: {avg_train_loss:.4f}")
        print(f"  • Val Loss: {avg_val_loss:.4f}")
        print(f"  • Val Accuracy: {accuracy:.2f}%")

        scheduler.step(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'class_to_idx': class_to_idx,
                'history': history
            }, model_path)
            print(f"  ✅ best model! ({accuracy:.2f}%)")

    # Fin de la boucle - attention à l'indentation!
    print("\n" + "="*60)
    print("✅ training ended!")
    print("="*60)
    print(f"\n🏆 best accuracy: {best_acc:.2f}%")
    print(f"💾 Model: {save_dir}/best_model.pth")

    # Courbes (optionnel)
    print("\n📊 Génération des courbes...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Courbes de Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(epochs_range, history['val_accuracy'], 'g-', label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title("Curve of Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"CPU Training - Best Acc: {best_acc:.2f}%")
    plt.tight_layout()
    plt.savefig('training_history_cpu.png')
    plt.show()

if __name__ == "__main__":
    main()