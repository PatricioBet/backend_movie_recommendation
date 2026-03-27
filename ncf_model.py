import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32, 16], dropout=0.2):
        super(NCF, self).__init__()
        
        self.user_embedding_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        self.user_embedding_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            input_size = layer_size
            
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        self.output_layer = nn.Linear(input_size + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        user_embedding_gmf = self.user_embedding_gmf(user_indices)
        item_embedding_gmf = self.item_embedding_gmf(item_indices)
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)
        
        user_embedding_mlp = self.user_embedding_mlp(user_indices)
        item_embedding_mlp = self.item_embedding_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        combined_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        prediction = self.output_layer(combined_vector)
        return self.sigmoid(prediction).squeeze()
