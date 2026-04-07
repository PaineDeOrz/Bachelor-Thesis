import torch
import torch.nn.functional as F
from utils import parse_args, get_all_possible_moves, create_elo_dict
from discriminator_model import Discriminator
from discriminator_dataset import DiscriminatorDataset

# Load your Maia2 config
cfg = parse_args('config.yaml')  # Your existing config file
all_moves = get_all_possible_moves()
all_moves_dict = {move: i for i, move in enumerate(all_moves)}
elo_dict = create_elo_dict()
num_moves = len(all_moves)

print(f"Config: dim_cnn={cfg.dim_cnn}, dim_vit={cfg.dim_vit}, input_channels={cfg.input_channels}")
print(f"Num moves: {num_moves}, Elo buckets: {len(elo_dict)}")

# Test 1: Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator = Discriminator(cfg, num_moves).to(device)
print(f"Discriminator created: {sum(p.numel() for p in discriminator.parameters())} params")

# Test 2: Create dummy batch (exact MAIA2Dataset shapes)
B = 32
boards = torch.randn(B, 18*8*8).to(device)  # [B, 1152] flattened
moves = torch.randint(0, num_moves, (B,)).to(device)
print(f"Test input shapes: boards={boards.shape}, moves={moves.shape}")

# Test 3: Forward pass
discriminator.eval()
with torch.no_grad():
    logits = discriminator(boards, moves)
    probs = torch.sigmoid(logits)
print(f"? Output shapes OK: logits={logits.shape}, probs={probs.shape}")
print(f"Sample logits: {logits[:5].cpu().tolist()}")
print(f"Sample probs:  {probs[:5].cpu().tolist()}")

# Test 4: With real CSV dataset
try:
    dataset = DiscriminatorDataset('dataset/discriminator_dataset.csv', all_moves_dict, elo_dict, cfg, max_samples=1000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    human_count, fake_count = 0, 0
    for boards_flat, moves, labels in loader:
        boards_flat, moves, labels = boards_flat.to(device), moves.to(device), labels.to(device)
        
        # Count labels
        human_count += (labels == 1.0).sum().item()
        fake_count += (labels == 0.0).sum().item()
        
        # Forward pass
        logits = discriminator(boards_flat, moves)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        print(f"Batch: human={human_count}, fake={fake_count}, loss={loss.item():.4f}")
        print(f"  Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"  Human probs:  [{probs[labels==1.0].mean():.3f} +- {probs[labels==1.0].std():.3f}]")
        print(f"  Fake probs:   [{probs[labels==0.0].mean():.3f} +- {probs[labels==0.0].std():.3f}]")
        break
    
    print(f"? CSV PROCESSING PERFECT: {human_count} human + {fake_count} fake samples")
except FileNotFoundError:
    print("Create dataset/discriminator_dataset.csv with columns: board,move,fake_move_1,fake_move_2,fake_move_3,elo_self,elo_oppo,active_win")
except KeyError as e:
    print(f"Missing CSV column: {e}")