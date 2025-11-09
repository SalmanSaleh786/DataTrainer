import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv  # ✅ Replaced GATConv with SAGEConv
import os, glob, ast, re
from collections import defaultdict
import random
import time

# ===========================
# 1. Load and preprocess data
# ===========================

def load_txt(file_path):
    with open(file_path, 'r') as f:
        allParsed = []
        for line in f:
            line = line.strip()
            firstIdx = line.find('(')
            if line.endswith(")"):
                line = line[firstIdx + 1:-1]
                data = ast.literal_eval(f"({line})")
                if len(data) != 11:
                    raise Exception("Length of data is not 11!")
                allParsed.append(data)
        return allParsed


def process_data(data_line):
    elements = data_line
    agentIndex, currDronePos, objectsAround, otherAgentPositions, wallCorners, battery, fire, foodCorners, score, action, nextPos = elements
    x_pos, y_pos = currDronePos
    obj_enc = [0 if obj in {'%', 'G', 'P'} else 1 if obj == 'F' else 0.5 for obj in objectsAround]
    walls_enc = [0 if obj else 1 for obj in wallCorners]
    food_enc = [1 if obj else 0 for obj in foodCorners]
    normalized_battery = float(battery) / 100.0
    normalized_score = float(score) / 10000
    normalized_x_pos = float(x_pos) / 25
    normalized_y_pos = float(y_pos) / 25

    node_features = torch.tensor([
        normalized_x_pos, normalized_y_pos,
        *obj_enc, *walls_enc, *food_enc,
        normalized_battery, normalized_score
    ], dtype=torch.float)

    next_x, next_y = nextPos
    normalized_next_x = float(next_x) / 25
    normalized_next_y = float(next_y) / 25
    next_pos_tensor = torch.tensor([normalized_next_x, normalized_next_y], dtype=torch.float)

    return node_features, next_pos_tensor


# ============================================
# 2. Load all graph data from mission log files
# ============================================

data_folder = "/home/salmansaleh/PycharmProjects/DataTrainer/logs/"
files = glob.glob(os.path.join(data_folder, "**", "*.txt"), recursive=True)

missionsByGameDict = defaultdict(dict)

for file in files:
    match = re.search(r"(\d+)_(Drone\d+)", file.split("/")[-1])
    if match:
        game_no = int(match.group(1))
        drone_id = int(match.group(2)[-1])
        mission_data = load_txt(file)
        missionsByGameDict[game_no][drone_id] = mission_data
    else:
        print('error filename:', file)

all_graphs = []
gameIdx = 0
for key in missionsByGameDict.keys():
    gameIdx += 1
    missionsOfThisGame = missionsByGameDict[key]
    max_steps = min(len(m) for m in missionsOfThisGame.values())
    previous_positions = {}

    for step in range(max_steps):
        nodes, edge_index, labels = [], [], []
        drone_positions = {}

        for drone_id, mission in missionsOfThisGame.items():
            if step >= len(mission):
                continue
            pos = mission[step][1]
            node_feat, next_pos = process_data(mission[step])
            node_index = len(nodes)
            nodes.append(node_feat)
            labels.append(next_pos)
            drone_positions[drone_id] = node_index

            if drone_id in previous_positions:
                edge_index.append([previous_positions[drone_id], node_index])
                edge_index.append([node_index, previous_positions[drone_id]])

        previous_positions = drone_positions.copy()

        # Connect nearby drones
        max_distance = 2
        drone_ids = list(drone_positions.keys())
        for i, drone_a in enumerate(drone_ids):
            for j, drone_b in enumerate(drone_ids):
                if i != j:
                    pos_a = missionsOfThisGame[drone_a][step][1]
                    pos_b = missionsOfThisGame[drone_b][step][1]
                    distance = abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
                    if distance <= max_distance:
                        edge_index.append([drone_positions[drone_a], drone_positions[drone_b]])
                        edge_index.append([drone_positions[drone_b], drone_positions[drone_a]])

        if not nodes:
            continue

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_features_tensor = torch.stack(nodes)
        labels_tensor = torch.stack(labels)
        graph_data = Data(x=node_features_tensor, edge_index=edge_index_tensor, y=labels_tensor)
        all_graphs.append(graph_data)

print(f"Total graphs created: {len(all_graphs)}")

# ===================================================
# 3. Split 80/20 for training and testing
# ===================================================

random.shuffle(all_graphs)
split_idx = int(0.8 * len(all_graphs))
train_graphs = all_graphs[:split_idx]
test_graphs = all_graphs[split_idx:]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

print(f"Training graphs: {len(train_graphs)}, Testing graphs: {len(test_graphs)}")

# ================================
# 4. Define a GraphSAGE Network
# ================================

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x


# ===============================
# 5. Training loop (same as before)
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_graphs[0].x.shape[1]
hidden_dim = 128
output_dim = 2

model = GraphSAGE(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.MSELoss()

for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation with timing
    model.eval()
    val_loss = 0
    total_inference_time = 0
    total_graphs = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            start_time = time.time()
            pred = model(batch.x, batch.edge_index)
            end_time = time.time()

            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_graphs += batch.num_graphs
            val_loss += criterion(pred, batch.y).item()

    val_loss /= len(test_loader)
    avg_inference_time = total_inference_time / total_graphs if total_graphs > 0 else 0

    print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Avg Inference: {avg_inference_time * 1000:.3f} ms/graph")

torch.save(model.state_dict(), "GraphSAGE_swarm_model.pt")
print("✅ Training complete. Model saved as GraphSAGE_swarm_model.pt")
