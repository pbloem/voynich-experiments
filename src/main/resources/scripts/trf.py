import former
from former.util import d, compute_compression, sample

import fire, tqdm, random, wandb

from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F


def estimate_compression(model, data, nsamples, context, batch_size, verbose=False, model_produces_logits=False):
    """
    Estimates the compression by sampling random subsequences instead of predicting all characters.

    NB: This doesn't work for GPT-2 style models with super-character tokenization, since the tokens and number of
    characters are mismatched.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.
    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        # current is the character to be predicted

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        target_indices.append(instance.size(0) - 2) # index of the last element of the context

        if instance.size(0) < context + 1:
            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[:, -1]  # target values
            # -- I think this is wrong?

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples # total nr of bits used


def sample_sequence(model, seed, max_context, length=600, temperature=1.0):
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """
    sequence = seed.detach().clone()

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    return sequence

def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target

def go(infile, rm_whitespace=True, tagged=False, trainprop=0.95, num_batches=100_000, batch_size=64, context=64,
       emb=256, layers=12, lr=3e-4, gradient_clipping=1.0, test_every=1000, lr_warmup=10_000, sample_length=128,
       seedlength=32, debug=False, name='vms-trf', project='vms-trf', valsamples=500):

    parms = locals()

    wd = wandb.init(
        name=name,
        project=project,
        config=parms,
        mode= 'disabled' if debug else 'online'
    )

    # Read file to corpus (strip whitespace)
    with open(infile, 'r') as file:
        all = file.read()

    if tagged:
        tokens = [token.split('.')[0] for token in all.split()]
    else:
        tokens = all.split()

    joinchar = '' if rm_whitespace else '_'
    corpus = joinchar.join(tokens)

    ctr = Counter(corpus)
    chars = [c for c, _ in ctr.most_common()]
    i2c, c2i = {i: c for i,c in enumerate(chars)}, {c:i for i,c in enumerate(chars)}
    print(f'Loaded corpus. {len(i2c)} characters found.')

    # -- sep (small) val set (10%)
    corpus = torch.tensor([c2i[c] for c in corpus], device=d())

    trainsize = int(len(corpus) * trainprop)
    train, val = corpus[:trainsize], corpus[trainsize:]
    if torch.cuda.is_available():
        train, val = train.cuda(), val.cuda()

    # Train model in random batches
    # create the model
    heads = emb//64
    model = former.GTransformer(emb=emb, heads=heads, depth=layers, seq_length=context,
                         num_tokens=len(i2c))
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / batch_size), 1.0))

    # Training loop
    instances_seen = 0
    for i in tqdm.trange(num_batches):

        opt.zero_grad()

        source, target = sample_batch(train, length=context, batch_size=batch_size)
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        output = model(source)  # forward pass

        # Compute the loss
        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        loss.backward()  # backward pass

        # clip gradients
        # -- If the total gradient vector has a length > x, we clip it back down to x.
        if gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        wandb.log({
            'loss': loss
        })

        opt.step()  # stochastic gradient descent step
        sch.step()  # update the learning rate

        if i != 0 and (i % test_every == 0 or i == num_batches - 1):
            with torch.no_grad():

                ## Sample and print a random sequence

                # Slice a random seed from the test data, and sample a continuation from the model.
                seedfr = random.randint(0, val.size(0) - seedlength)
                seed = val[seedfr:seedfr + seedlength].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                seq = sample_sequence(model, seed=seed, max_context=context, length=sample_length)
                seq = ''.join(i2c[i.item()] for i in seq)
                print(seq[:seedlength], seq[seedlength:])

                ## Compute validation bits per byte
                bits_per_byte = estimate_compression(model, val, context=context, batch_size=batch_size * 2, nsamples=valsamples)
                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')

                wandb.log({
                    'val': bits_per_byte
                })

    # Save model and char dicts
    torch.save({
        'model' : model.state_dict(),
        'parms' : parms,
        'i2c' : i2c,
        'c2i' : c2i
    }, 'model.cpt')

    # Tokenize
    tokenize(model, corpus, i2c, c2i)

def tokenize(model, corpus=None, i2c=None, c2i=None, outfile='tokenized.txt', infile=None):

    if type(model is str):
        pass
        # load model and dicts

    if corpus is None:
        pass

    # Compute all entropies on character boundaries


    # Make plots

    # Tokenize the corpus


if __name__ == '__main__':
    fire.Fire()