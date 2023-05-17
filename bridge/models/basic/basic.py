import torch
from .layers import MLP
from .time_embedding import get_timestep_embedding


class ScoreNetwork(torch.nn.Module):

    def __init__(self, time_emb_dim=32, encoder_layers=[128, 256], decoder_layers=[256, 128], x_dim=2):
        super().__init__()
        self.temb_dim = time_emb_dim
        t_enc_dim = encoder_layers[-1]
        self.locals = [encoder_layers, time_emb_dim, decoder_layers, x_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [x_dim],
                       activate_final=False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(time_emb_dim,
                             layer_widths=encoder_layers,
                             activate_final=False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers,
                             activate_final=False,
                             activation_fn=torch.nn.LeakyReLU())

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out
