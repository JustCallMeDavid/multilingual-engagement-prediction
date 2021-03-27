import torch
import torch.nn

class Char(torch.nn.Module):

    def __init__(self, args):
        super(Char, self).__init__()
        self.args = args

        # we have unicode characters with ids between 0 and 65536 in our dataset
        self.emb_layer = torch.nn.Embedding(65536, args.emb_dim)

        self.first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=args.ch_in,
                            out_channels=args.ch_out,

                            kernel_size=(1, args.emb_dim), stride=1),
            torch.nn.ReLU()
        )

        self.convolutions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=args.ch_in,
                                out_channels=args.ch_out,

                                kernel_size=(1, args.ch_out)),
                torch.nn.ReLU()) for i in range(args.number_convolutions)])

        self.first_forward_layer = torch.nn.Sequential(
            torch.nn.Linear(2048, args.forward_layer_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=args.dropout))

        self.forward_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(args.forward_layer_sizes[i], args.forward_layer_sizes[i + 1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=args.dropout))
            for i in range(len(args.forward_layer_sizes) - 1)])

        self.final_forward = torch.nn.Linear(args.forward_layer_sizes[-1], args.number_classes)

    def forward(self, x):

        x = self.emb_layer(x)
        x = x.unsqueeze(dim=1)
        x = self.first_conv(x)

        for conv in self.convolutions:
            x = x.transpose(1, 3)
            x = torch.nn.functional.max_pool2d(x, kernel_size=self.args.kernel_size_pool, stride=self.args.stride_pool)
            x = conv(x)

        x = x.view(x.size(0), -1)
        x = self.first_forward_layer(x)

        for forward_layer in self.forward_layers:
            x = forward_layer(x)

        x = self.final_forward(x)
        # do not convert to softmax inside network as torch CrossEntropyLoss expects logits (and calls softmax itself)
        return x.squeeze()