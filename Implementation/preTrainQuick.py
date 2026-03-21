import logging
import torch
import preTrainUtils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler


try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter



def quickTest(args, train_dataset, model, tokenizer, i=0):
    def collate_fn(batch):
        # is given to dataloader in order to pad batches on the go
        # rather than all at once
        return pad_sequence(
            batch,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
    )
    batch = next(iter(train_dataloader))

    vocab_items = list(tokenizer.vocab.items())

    # 3. Check what ID it gives when forced
    print('Before conversion to tokens: ', batch[1][i:i+10])
    print('Before mask: ', tokenizer.convert_ids_to_tokens(batch[1][i:i+10]))
    inputs, labels = preTrainUtils.mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
    # Create attention mask for pad tokens (1 for real tokens, 0 for pad tokens)
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    print('After mask: ', tokenizer.convert_ids_to_tokens(inputs[1][i:i+10]))

    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    attention_mask = attention_mask.to(args.device)    
    outputs = model(inputs, attention_mask=attention_mask, labels=labels) if args.mlm else model(inputs, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    converted_predictions = [tokenizer.convert_ids_to_tokens(seq) for seq in predictions]
    print('Model: ', converted_predictions[1][i:i+10])



def main():
    args, tokenizer, model = preTrainUtils.param_setup()

    if args.pcap_folder:
        preTrainUtils.readPcap_folder(args.pcap_folder, args.pcap_out)
    
    train_dataset = preTrainUtils.load_and_cache_examples(args, tokenizer)
    if args.quick_test == True:
        quickTest(args, train_dataset, model, tokenizer, 10)    
    if args.do_train == True:
        preTrainUtils.train(args, train_dataset, model, tokenizer)
    if args.eval == True:
        print(preTrainUtils.evaluate(args, model, tokenizer))
    return


if __name__ == '__main__':
    main()
