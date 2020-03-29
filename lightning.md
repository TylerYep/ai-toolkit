Model Verification in Trainer

## 🚀 Feature
<!-- A clear and concise description of the feature proposal -->

Verifies that the provided model code does not mix up data across the batch dimension. We do this by setting the loss to be something trivial (e.g. the sum of all outputs of example i), running the backward pass all the way to the input, and ensuring that we only get a non-zero gradient on the i-th input.


### Motivation
<!-- Please outline the motivation for the proposal. Is your feature request related to a problem? e.g., I'm always frustrated when [...]. If this is related to another GitHub issue, please link here too -->

First of all, I would like to say thank you for the fantastic work being done on this project. Recently, I was working on a side project that has almost the exact same goal as this one, which I used as motivation to learn more about PyTorch and how to make Deep Learning easier. Clearly, this project is a lot more thought-out than mine :^), but I wanted to see if there were any ideas I developed independently that might be useful in this project.

One of the most useful utils I've implemented is a verification step before the model runs. In my project, this verification step performs checks such as:
- ensuring data is not mixed across the batch dimension
- ensuring the model can overfit a single example
- ensuring that all layers of the model are training (or selected layers are properly frozen)

Since I am very new to this project, I thought that the first bullet point might be a good place to start.


### Pitch
<!-- A clear & concise description of what you want to happen. -->

Given the introductory example in the documention, assume we had written some poor tensor operations in our model like so:

```python
class BadModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        ###
        # x = x.view(batch_size, -1)
        ###
        x = x.view(-1, 1, 56, 56)
        x = x.permute(1, 0, 3, 2)
        x = x.reshape((batch_size, -1))
        ###

        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x
```

When we start to train our model, everything begins training smoothly. However, this code is clearly wrong - we are crossing image data from separate datapoints in our batch.

It would be helpful if Lightning gave us a warning if this has happened. For example:

```python
def check_batch_dimension(model, loader, optimizer, test_val=2):
    model.eval()
    torch.set_grad_enabled(True)
    data, _ = next(iter(loader))
    optimizer.zero_grad()
    data.requires_grad_()

    output = model(data)
    loss = output[test_val].sum()
    loss.backward()

    error_msg = "Your model is mixing up data across the batch dimension!"
    assert loss != 0
    assert (data.grad[test_val] != 0).any(), error_msg
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all(), error_msg
```

This function verifies that only a single datapoint in the batch should have a nonzero gradient. This check has saved me countless times from running a poorly written model. :)

Implementation-wise, I am looking for any advice on whether this is a useful effort, whether it fits into the intended goals of Lightning, and what are possible difficulties that may arise.


### Alternatives
<!-- A clear & concise description of any alternative solutions or features you've considered, if any. -->

It is clear that the feature as it stands will not work for all models, as some variants of LSTMs and such use a different dimension as its batch dimension. There also might be issues if the batch is split up as it accumulates - I'm not quite certain how everything in this project works.

However, I would expect that this would be useful in almost all models. I advocate this being a default warning, but also allowing well-intentioned users to simply pass some sort of flag to disable this verification step.

I also realize there needs to be some cleanup after this step to reset the model to its previous state. Any insights here would be great as well.


### Additional context
<!-- Add any other context or screenshots about the feature request here. -->
